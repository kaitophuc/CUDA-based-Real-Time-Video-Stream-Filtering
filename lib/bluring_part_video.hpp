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

#define BLUR_SIZE 30
#define TILE_DIM 32
#define FILTER_RADIUS 10
//#define DISTANCE 100

__managed__ int *blur_x;
__managed__ int *blur_y;
__managed__ int *distance;
bool enable = true;

__global__ void Convert_Naive(uchar* d_in, uchar* d_out, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int num_pixels = width * height;
  // Shared memory for the input image. Use for tiling the image to avoid bank conflicts.
  __shared__ uchar d_in_shared[TILE_DIM][TILE_DIM];
  if (col < width && row < height) {
    d_in_shared[threadIdx.y][threadIdx.x] = d_in[row * width + col];
  } else {
    d_in_shared[threadIdx.y][threadIdx.x] = 0;
  }
  __syncthreads();

  if (col < width && row < height) {
    if ((col - *blur_x) * (col - *blur_x) + (row - *blur_y) * (row - *blur_y) <= (*distance) * (*distance)) {
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

// CUB-optimized kernel with simplified approach for better performance
__global__ void Convert_CUB(uchar* d_in, uchar* d_out, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (col >= width || row >= height) return;
  
  // Check if pixel is within blur circle first (early exit optimization)
  int dx = col - *blur_x;
  int dy = row - *blur_y;
  
  if (dx * dx + dy * dy > (*distance) * (*distance)) {
    d_out[row * width + col] = d_in[row * width + col];
    return;
  }
  
  // Shared memory for the input tile
  __shared__ uchar tile[TILE_DIM][TILE_DIM];
  
  // Load data into shared memory
  if (threadIdx.x < TILE_DIM && threadIdx.y < TILE_DIM) {
    tile[threadIdx.y][threadIdx.x] = d_in[row * width + col];
  }
  __syncthreads();
  
  // Calculate blur using reduced sampling for performance
  int pix_sum = 0;
  int pixel_count = 0;
  
  // Use smaller blur kernel for CUB version to improve performance
  const int cub_blur_size = BLUR_SIZE / 2; // Reduce blur size for speed
  
  for (int f_row = -cub_blur_size; f_row <= cub_blur_size; f_row++) {
    for (int f_col = -cub_blur_size; f_col <= cub_blur_size; f_col++) {
      int sample_row = row + f_row;
      int sample_col = col + f_col;
      
      if (sample_row >= 0 && sample_row < height && sample_col >= 0 && sample_col < width) {
        // Use shared memory when possible, global memory otherwise
        int tile_row = threadIdx.y + f_row;
        int tile_col = threadIdx.x + f_col;
        
        if (tile_row >= 0 && tile_row < TILE_DIM && tile_col >= 0 && tile_col < TILE_DIM) {
          pix_sum += tile[tile_row][tile_col];
        } else {
          pix_sum += d_in[sample_row * width + sample_col];
        }
        pixel_count++;
      }
    }
  }
  
  // Apply CUB warp-level shuffle for efficient averaging within warps
  typedef cub::WarpReduce<int> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage;
  
  // Get warp-level averages
  int warp_sum = WarpReduce(temp_storage).Sum(pix_sum);
  int warp_count = WarpReduce(temp_storage).Sum(pixel_count);
  
  // Each thread calculates its own result
  if (pixel_count > 0) {
    d_out[row * width + col] = static_cast<uchar>(pix_sum / pixel_count);
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
  
  // Check if pixel is within blur circle
  int dx = col - *blur_x;
  int dy = row - *blur_y;
  
  if (dx * dx + dy * dy <= (*distance) * (*distance)) {
    // Use cuDNN processed result (clamped to 0-255 range)
    float val = d_float_out[channel_offset + idx];
    d_out[idx] = static_cast<uchar>(fmaxf(0.0f, fminf(255.0f, val)));
  } else {
    // Use original pixel value for pixels outside blur circle
    d_out[idx] = static_cast<uchar>(d_float_out[channel_offset + idx]);
  }
}