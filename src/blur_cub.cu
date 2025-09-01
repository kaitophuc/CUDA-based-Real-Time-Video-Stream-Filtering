#include "blur_cub.hpp"

//==========================================================================//
// CUB-optimized kernel with simplified approach for better performance
template<int R, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BoxBlurHorizontal(
    const uchar* d_in,
    int width, int height, int pitch,
    int *hsum)
{
  using BlockLoad = cub::BlockLoad<uchar, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE>;
  using BlockStore = cub::BlockStore<int, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_VECTORIZE>;
  using BlockScan = cub::BlockScan<int, BLOCK_THREADS>;

  constexpr int TILE = BLOCK_THREADS * ITEMS_PER_THREAD;
  int row  = blockIdx.y;
  if (row >= height) return;
  int x0 = blockIdx.x * TILE;
  int valid = max(0, min(TILE, width - x0));

  __shared__ union {
    typename BlockLoad::TempStorage load;
    typename BlockStore::TempStorage store;
    typename BlockScan::TempStorage scan;
  } temp_storage;

  uchar in_items[ITEMS_PER_THREAD];
  const uchar* row_ptr = d_in + row * pitch + x0;
  BlockLoad(temp_storage.load).Load(row_ptr, in_items, valid);
  __syncthreads();

  int vals[ITEMS_PER_THREAD];
  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int x = threadIdx.x * ITEMS_PER_THREAD + i;
    vals[i] = (x < valid) ? int(in_items[i]) : 0;
  }
  BlockScan(temp_storage.scan).InclusiveSum(vals, vals);
  __syncthreads();

  __shared__ int prefix[TILE + 1];
  if (threadIdx.x == 0) prefix[0] = 0;
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int x = threadIdx.x * ITEMS_PER_THREAD + i;
    if (x < valid) {
      prefix[x + 1] = vals[i];
    }
  }
  __syncthreads();

  int sum_out[ITEMS_PER_THREAD];
  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int x_local = threadIdx.x * ITEMS_PER_THREAD + i;
    int x_global = x0 + x_local;
    if (x_local < valid) {
      int left = max(0, x_global - R);
      int right = min(width - 1, x_global + R);
      int L = max(0, left - x0);
      int R_ = min(valid - 1, right - x0);
      int sum = 0;
      if (L <= R_) {
        sum = prefix[R_ + 1] - prefix[L];
      } 
      for (int c = left; c < x0; c++) {
        sum += d_in[row * pitch + c];
      }
      for (int c = x0 + valid; c <= right; c++) {
        if (c < width) {
          sum += d_in[row * pitch + c];
        }
      }
      sum_out[i] = sum;
    }
  }
  BlockStore(temp_storage.store).Store(hsum + row * pitch + x0, sum_out, valid);
}

template<int R, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BoxBlurVertical(
    const int* hsum,
    const uchar* d_in,
    uchar* d_out,
    int width, int height, int pitch)
{
  using BlockScan = cub::BlockScan<int, BLOCK_THREADS>;
  constexpr int TILE = BLOCK_THREADS * ITEMS_PER_THREAD;

  int col = blockIdx.x;
  int y0 = blockIdx.y * TILE;
  int valid = max(0, min(TILE, height - y0));
  if (col >= width) return;

  __shared__ typename BlockScan::TempStorage scan1;
  __shared__ int prefix_sum[TILE + 1];
  if (threadIdx.x == 0) {
    prefix_sum[0] = 0;
  }
  __syncthreads();

  int vals_sum[ITEMS_PER_THREAD];
  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int y_local = threadIdx.x * ITEMS_PER_THREAD + i;
    int y_global = y0 + y_local;
    if (y_local < valid) {
      vals_sum[i] = hsum[y_global * pitch + col];
    }
    else {
      vals_sum[i] = 0;
    }
  }
  BlockScan(scan1).InclusiveSum(vals_sum, vals_sum);
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int y_local = threadIdx.x * ITEMS_PER_THREAD + i;
    if (y_local < valid) {
      prefix_sum[y_local + 1] = vals_sum[i];
    }
  }
  __syncthreads();

  int nfaces = min(*num_faces, MAX_FACES);
  
  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int y_local = threadIdx.x * ITEMS_PER_THREAD + i;
    int y_global = y0 + y_local;
    if (y_local < valid) {
      int top = max(0, y_global - R);
      int bottom = min(height - 1, y_global + R);
      int T = max(0, top - y0);
      int B = min(valid - 1, bottom - y0);
      int sum = 0;
      if (T <= B) {
        sum = prefix_sum[B + 1] - prefix_sum[T];
      }
      for (int r = top; r < y0; r++) {
        sum += hsum[r * pitch + col];
      }
      for (int r = y0 + valid; r <= bottom; r++) {
        if (r < height) {
          sum += hsum[r * pitch + col];
        }
      }
      
      // Use box_count function to calculate the number of pixels in the blur kernel
      int count = box_count(col, y_global, width, height, R);
      int avg = (count > 0) ? (sum / count) : int(d_in[y_global * pitch + col]);

      bool should_blur = false;
      if (nfaces == 1) {
        int dx = col - blur_x[0];
        int dy = y_global - blur_y[0];
        should_blur = (dx * dx + dy * dy <= distance[0] * distance[0]);
      } else {
        #pragma unroll
        for (int face_idx = 0; face_idx < nfaces; face_idx++) {
          int dx = col - blur_x[face_idx];
          int dy = y_global - blur_y[face_idx];
          if (dx * dx + dy * dy <= distance[face_idx] * distance[face_idx]) {
            should_blur = true;
            break;
          }
        }
      }
      d_out[y_global * pitch + col] = should_blur ? static_cast<uchar>(avg) : d_in[y_global * pitch + col];
    }
  }
}

//==========================================================================//

// CUB-optimized kernel with advanced block-level reductions
void Blur_CUB(cv::Mat& frame, int width, int height, int frames, int num_pixels, uchar* hr_in, uchar* hg_in, uchar* hb_in, 
              uchar* hr_out, uchar* hg_out, uchar* hb_out, uchar* dr_in, uchar* dg_in, uchar* db_in, 
              uchar* dr_out, uchar* dg_out, uchar* db_out) {

  cudaStream_t streams[3];
  for (int i = 0; i < 3; i++) {
    cudaStreamCreate(&streams[i]);
  }

  const int W = width;
  const int H = height;
  const int P = width;
  size_t bytesI = W * H * sizeof(int);

  // Only allocate sum arrays, no need for count arrays since we use box_count function
  int *d_hsum_r, *d_hsum_g, *d_hsum_b;
  cudaMalloc(&d_hsum_r, bytesI);
  cudaMalloc(&d_hsum_g, bytesI);
  cudaMalloc(&d_hsum_b, bytesI);
  
  dim3 bh(128), gv(W, (H + bh.x * 8 - 1) / (bh.x * 8));
  dim3 gh((W + (bh.x * 8 - 1)) / (bh.x * 8), H);
  
  // Asynchronous operations for RGB channels using CUB kernel
  cudaMemcpyAsync(dr_in, hr_in, num_pixels * sizeof(uchar), cudaMemcpyHostToDevice, streams[0]);
  cudaMemcpyAsync(dg_in, hg_in, num_pixels * sizeof(uchar), cudaMemcpyHostToDevice, streams[1]);
  cudaMemcpyAsync(db_in, hb_in, num_pixels * sizeof(uchar), cudaMemcpyHostToDevice, streams[2]);

  // Horizontal pass - compute row sums only
  BoxBlurHorizontal<BLUR_SIZE><<<gh, bh, 0, streams[0]>>>(dr_in, W, H, P, d_hsum_r);
  BoxBlurHorizontal<BLUR_SIZE><<<gh, bh, 0, streams[1]>>>(dg_in, W, H, P, d_hsum_g);
  BoxBlurHorizontal<BLUR_SIZE><<<gh, bh, 0, streams[2]>>>(db_in, W, H, P, d_hsum_b);

  // Vertical pass - compute final blur using box_count function for pixel counts
  BoxBlurVertical<BLUR_SIZE><<<gv, bh, 0, streams[0]>>>(d_hsum_r, dr_in, dr_out, W, H, P);
  BoxBlurVertical<BLUR_SIZE><<<gv, bh, 0, streams[1]>>>(d_hsum_g, dg_in, dg_out, W, H, P);
  BoxBlurVertical<BLUR_SIZE><<<gv, bh, 0, streams[2]>>>(d_hsum_b, db_in, db_out, W, H, P);

  cudaMemcpyAsync(hr_out, dr_out, num_pixels * sizeof(uchar), cudaMemcpyDeviceToHost, streams[0]);
  cudaMemcpyAsync(hg_out, dg_out, num_pixels * sizeof(uchar), cudaMemcpyDeviceToHost, streams[1]);
  cudaMemcpyAsync(hb_out, db_out, num_pixels * sizeof(uchar), cudaMemcpyDeviceToHost, streams[2]);

  // Synchronize all streams
  for (int i = 0; i < 3; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  // Free temporary buffers (only sum arrays now)
  cudaFree(d_hsum_r);
  cudaFree(d_hsum_g);
  cudaFree(d_hsum_b);

  // Update frame data
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      cv::Vec3b& pixel = frame.at<cv::Vec3b>(i, j);
      pixel[2] = hr_out[i * width + j];
      pixel[1] = hg_out[i * width + j];
      pixel[0] = hb_out[i * width + j];
    }
  }
}

// Explicit template instantiation to avoid linking issues
template __global__ void BoxBlurHorizontal<BLUR_SIZE, 128, 8>(const uchar*, int, int, int, int*);
template __global__ void BoxBlurVertical<BLUR_SIZE, 128, 8>(const int*, const uchar*, uchar*, int, int, int);
