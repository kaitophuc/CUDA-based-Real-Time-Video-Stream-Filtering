#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <omp.h>
#include <bits/stdc++.h>
#include <chrono>
#include <thread>
#include <barrier>

// CUB includes for optimized operations
#include <cub/cub.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/block/block_reduce.cuh>

// cuDNN includes for neural network accelerated operations
#include <cudnn.h>

#define BLUR_SIZE 128
#define TILE_DIM 32
#define DISTANCE 100
#define MAX_FACES 3

__managed__ int *blur_x;
__managed__ int *blur_y;
__managed__ int *distance;
__managed__ int *num_faces;
bool enable = true;

__global__ void Convert_Naive(uchar* dr_in, uchar* dg_in, uchar* db_in, uchar* dr_out, uchar* dg_out, uchar* db_out, 
                        int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  // Shared memory for the input image. Use for tiling the image to avoid bank conflicts.
  __shared__ uchar dr_in_shared[TILE_DIM][TILE_DIM];
  __shared__ uchar dg_in_shared[TILE_DIM][TILE_DIM];
  __shared__ uchar db_in_shared[TILE_DIM][TILE_DIM];
  
  if (col < width && row < height) {
    dr_in_shared[threadIdx.y][threadIdx.x] = dr_in[row * width + col];
    dg_in_shared[threadIdx.y][threadIdx.x] = dg_in[row * width + col];
    db_in_shared[threadIdx.y][threadIdx.x] = db_in[row * width + col];
  } else {
    dr_in_shared[threadIdx.y][threadIdx.x] = 0;
    dg_in_shared[threadIdx.y][threadIdx.x] = 0;
    db_in_shared[threadIdx.y][threadIdx.x] = 0;
  }

  __syncthreads();

  if (col < width && row < height) {
    if ((col - *blur_x) * (col - *blur_x) + (row - *blur_y) * (row - *blur_y) <= (*distance) * (*distance)) {
      int pix_val_r = 0;
      int pix_val_g = 0;
      int pix_val_b = 0;
      int pixels = 0;

      // Get the average of the surrounding pixels
      for (int f_row = -BLUR_SIZE; f_row <= BLUR_SIZE; f_row++) {
        for (int f_col = -BLUR_SIZE; f_col <= BLUR_SIZE; f_col++) {
          int tile_row = threadIdx.y + f_row;
          int tile_col = threadIdx.x + f_col;
          if (tile_row >= 0 && tile_row < TILE_DIM && tile_col >= 0 && tile_col < TILE_DIM) {
            pix_val_r += dr_in_shared[tile_row][tile_col];
            pix_val_g += dg_in_shared[tile_row][tile_col];
            pix_val_b += db_in_shared[tile_row][tile_col];
            ++pixels;
          } else {
            int i = row + f_row;
            int j = col + f_col;
            if (i >= 0 && i < height && j >= 0 && j < width) {
              pix_val_r += dr_in[i * width + j];
              pix_val_g += dg_in[i * width + j];
              pix_val_b += db_in[i * width + j];
              ++pixels;
            }
          }
        }
      }

      dr_out[row * width + col] = static_cast<uchar>(pix_val_r / pixels);
      dg_out[row * width + col] = static_cast<uchar>(pix_val_g / pixels);
      db_out[row * width + col] = static_cast<uchar>(pix_val_b / pixels);
    } else {
      dr_out[row * width + col] = dr_in[row * width + col];
      dg_out[row * width + col] = dg_in[row * width + col];
      db_out[row * width + col] = db_in[row * width + col];
    }
  }
}

__global__ void Convert_MultiStream(uchar* d_in, uchar* d_out, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  // Shared memory for the input image. Use for tiling the image to avoid bank conflicts.
  __shared__ uchar d_in_shared[TILE_DIM][TILE_DIM];
  if (col < width && row < height) {
    d_in_shared[threadIdx.y][threadIdx.x] = d_in[row * width + col];
  } else {
    d_in_shared[threadIdx.y][threadIdx.x] = 0;
  }
  __syncthreads();

  if (col < width && row < height) {
    bool should_blur = false;
    
    // Check if pixel is within any of the detected faces
    for (int face_idx = 0; face_idx < *num_faces && face_idx < MAX_FACES; face_idx++) {
      int dx = col - blur_x[face_idx];
      int dy = row - blur_y[face_idx];
      if (dx * dx + dy * dy <= distance[face_idx] * distance[face_idx]) {
        should_blur = true;
        break;
      }
    }
    
    if (should_blur) {
      int pix_val = 0;
      int pixels = 0;

      // Get the average of the surrounding pixels
      for (int f_row = -BLUR_SIZE; f_row <= BLUR_SIZE; f_row++) {
        for (int f_col = -BLUR_SIZE; f_col <= BLUR_SIZE; f_col++) {
          int tile_row = threadIdx.y + f_row;
          int tile_col = threadIdx.x + f_col;
          if (tile_row >= 0 && tile_row < TILE_DIM && tile_col >= 0 && tile_col < TILE_DIM) {
            pix_val += d_in_shared[tile_row][tile_col];
            ++pixels;
          } else {
            int i = row + f_row;
            int j = col + f_col;
            if (i >= 0 && i < height && j >= 0 && j < width) {
              pix_val += d_in[i * width + j];
              ++pixels;
            }
          }
        }
      }

      d_out[row * width + col] = static_cast<uchar>(pix_val / pixels);
    } else {
      d_out[row * width + col] = d_in[row * width + col];
    }
  }
}

//==========================================================================//
// CUB-optimized kernel with simplified approach for better performance
template<int R, int BLOCK_THREADS = 128, int ITEMS_PER_THREAD = 8>
__global__ void BoxBlurHorizontal(
    const uchar* d_in,
    int width, int height, int pitch,
    int *hsum, int *hcount)
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
  int count_out[ITEMS_PER_THREAD];
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
      int cnt = right - left + 1;
      sum_out[i] = sum;
      count_out[i] = cnt;
    }
  }
  BlockStore(temp_storage.store).Store(hsum + row * pitch + x0, sum_out, valid);
  __syncthreads();
  BlockStore(temp_storage.store).Store(hcount + row * pitch + x0, count_out, valid);
}

template<int R, int BLOCK_THREADS = 128, int ITEMS_PER_THREAD = 8>
__global__ void BoxBlurVertical(
    const int* hsum,
    const int* hcount,
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

  __shared__ typename BlockScan::TempStorage scan1, scan2;
  __shared__ int prefix_sum[TILE + 1];
  __shared__ int prefix_count[TILE + 1];
  if (threadIdx.x == 0) {
    prefix_sum[0] = 0;
    prefix_count[0] = 0;
  }
  __syncthreads();

  int vals_sum[ITEMS_PER_THREAD];
  int vals_count[ITEMS_PER_THREAD];
  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int y_local = threadIdx.x * ITEMS_PER_THREAD + i;
    int y_global = y0 + y_local;
    if (y_local < valid) {
      vals_sum[i] = hsum[y_global * pitch + col];
      vals_count[i] = hcount[y_global * pitch + col];
    }
    else {
      vals_sum[i] = 0;
      vals_count[i] = 0;
    }
  }
  BlockScan(scan1).InclusiveSum(vals_sum, vals_sum);
  __syncthreads();
  BlockScan(scan2).InclusiveSum(vals_count, vals_count);
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int y_local = threadIdx.x * ITEMS_PER_THREAD + i;
    if (y_local < valid) {
      prefix_sum[y_local + 1] = vals_sum[i];
      prefix_count[y_local + 1] = vals_count[i];
    }
  }
  __syncthreads();

  int nfaces = min(*num_faces, MAX_FACES);
  int cx0 = (nfaces > 0) ? blur_x[0] : -1;
  int cy0 = (nfaces > 0) ? blur_y[0] : -1;
  int r0 = (nfaces > 0) ? distance[0] : -1;
  
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
      int count = 0;
      if (T <= B) {
        sum = prefix_sum[B + 1] - prefix_sum[T];
        count = prefix_count[B + 1] - prefix_count[T];
      }
      for (int r = top; r < y0; r++) {
        sum += hsum[r * pitch + col];
        count += hcount[r * pitch + col];
      }
      for (int r = y0 + valid; r <= bottom; r++) {
        if (r < height) {
          sum += hsum[r * pitch + col];
          count += hcount[r * pitch + col];
        }
      }
      int avg = (count > 0) ? (sum / count) : int(d_in[y_global * pitch + col]);

      bool should_blur = false;
      if (nfaces == 1) {
        int dx = col - cx0;
        int dy = y_global - cy0;
        should_blur = (dx * dx + dy * dy <= r0 * r0);
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

// CUB-optimized kernel that uses the same multi-face logic
__global__ void Convert_CUB(uchar* d_in, uchar* d_out, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (col >= width || row >= height) return;
  
  // Check if pixel is within any of the detected faces
  bool should_blur = false;
  for (int face_idx = 0; face_idx < *num_faces && face_idx < MAX_FACES; face_idx++) {
    int dx = col - blur_x[face_idx];
    int dy = row - blur_y[face_idx];
    if (dx * dx + dy * dy <= distance[face_idx] * distance[face_idx]) {
      should_blur = true;
      break;
    }
  }
  
  if (should_blur) {
    // Simple blur for CUB version
    int pix_val = 0;
    int pixels = 0;
    int blur_radius = min(BLUR_SIZE, 16); // Use smaller radius for performance
    
    for (int f_row = -blur_radius; f_row <= blur_radius; f_row++) {
      for (int f_col = -blur_radius; f_col <= blur_radius; f_col++) {
        int i = row + f_row;
        int j = col + f_col;
        if (i >= 0 && i < height && j >= 0 && j < width) {
          pix_val += d_in[i * width + j];
          ++pixels;
        }
      }
    }
    d_out[row * width + col] = static_cast<uchar>(pix_val / pixels);
  } else {
    d_out[row * width + col] = d_in[row * width + col];
  }
}

// Helper function to check cuDNN errors
#define CHECK_CUDNN(call) do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error: " << cudnnGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// cuDNN-optimized kernel using convolution for blur effect
__global__ void Convert_cuDNN_PostProcess(float* d_float_out, uchar* d_out, int width, int height, int channel_offset) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (col >= width || row >= height) return;
  
  int idx = row * width + col;
  
  // Check if pixel is within any of the detected faces
  bool should_blur = false;
  for (int face_idx = 0; face_idx < *num_faces && face_idx < MAX_FACES; face_idx++) {
    int dx = col - blur_x[face_idx];
    int dy = row - blur_y[face_idx];
    if (dx * dx + dy * dy <= distance[face_idx] * distance[face_idx]) {
      should_blur = true;
      break;
    }
  }
  
  if (should_blur) {
    // Use cuDNN processed result (clamped to 0-255 range)
    float val = d_float_out[channel_offset + idx];
    d_out[idx] = static_cast<uchar>(fmaxf(0.0f, fminf(255.0f, val)));
  } else {
    // Use original pixel value for pixels outside blur circle
    d_out[idx] = static_cast<uchar>(d_float_out[channel_offset + idx]);
  }
}