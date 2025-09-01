#include "blur_multistream.hpp"

// Multi-stream CUDA kernel implementation - supports multiple faces with separate streams
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

// Optimized kernel with shared memory. Somehow, the performance is even better than 
// the naive version with overlapped memory access and compute.
void Blur_MultiStream(cv::Mat& frame, int width, int height, int frames, int num_pixels, uchar* hr_in, uchar* hg_in, uchar* hb_in, 
                   uchar* hr_out, uchar* hg_out, uchar* hb_out, uchar* dr_in, uchar* dg_in, uchar* db_in, 
                   uchar* dr_out, uchar* dg_out, uchar* db_out) {

  cudaStream_t streams[3];
  for (int i = 0; i < 3; i++) {
    cudaStreamCreate(&streams[i]);
  }

  dim3 block_size(TILE_DIM, TILE_DIM);
  dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

  // Asynchronous operations for RGB channels
  cudaMemcpyAsync(dr_in, hr_in, num_pixels * sizeof(uchar), cudaMemcpyHostToDevice, streams[0]);
  cudaMemcpyAsync(dg_in, hg_in, num_pixels * sizeof(uchar), cudaMemcpyHostToDevice, streams[1]);
  cudaMemcpyAsync(db_in, hb_in, num_pixels * sizeof(uchar), cudaMemcpyHostToDevice, streams[2]);

  Convert_MultiStream<<<grid_size, block_size, 0, streams[0]>>>(dr_in, dr_out, width, height);
  Convert_MultiStream<<<grid_size, block_size, 0, streams[1]>>>(dg_in, dg_out, width, height);
  Convert_MultiStream<<<grid_size, block_size, 0, streams[2]>>>(db_in, db_out, width, height);

  cudaMemcpyAsync(hr_out, dr_out, num_pixels * sizeof(uchar), cudaMemcpyDeviceToHost, streams[0]);
  cudaMemcpyAsync(hg_out, dg_out, num_pixels * sizeof(uchar), cudaMemcpyDeviceToHost, streams[1]);
  cudaMemcpyAsync(hb_out, db_out, num_pixels * sizeof(uchar), cudaMemcpyDeviceToHost, streams[2]);

  // Synchronize all streams
  for (int i = 0; i < 3; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

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
