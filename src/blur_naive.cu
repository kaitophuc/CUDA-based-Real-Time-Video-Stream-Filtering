#include "blur_naive.hpp"

// Naive CUDA kernel implementation - processes all RGB channels in a single kernel launch
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
    // Check if pixel is within any of the detected faces (consistent with other kernels)
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

// Naive kernel that processes all RGB channels in a single kernel launch
// Uses sequential memory transfers but only one kernel launch
void Blur_Naive(cv::Mat& frame, int width, int height, int frames, int num_pixels, uchar* hr_in, uchar* hg_in, uchar* hb_in, 
           uchar* hr_out, uchar* hg_out, uchar* hb_out, uchar* dr_in, uchar* dg_in, uchar* db_in, 
           uchar* dr_out, uchar* dg_out, uchar* db_out, cudaStream_t* streams) {

  // Use the first stream for naive implementation
  cudaStream_t stream = streams[0];

  dim3 block_size(TILE_DIM, TILE_DIM);
  dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

  // Transfer all RGB channels to device
  cudaMemcpyAsync(dr_in, hr_in, num_pixels * sizeof(uchar), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(dg_in, hg_in, num_pixels * sizeof(uchar), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(db_in, hb_in, num_pixels * sizeof(uchar), cudaMemcpyHostToDevice, stream);

  // Single kernel launch processes all RGB channels together
  Convert_Naive<<<grid_size, block_size, 0, stream>>>(dr_in, dg_in, db_in, dr_out, dg_out, db_out, width, height);

  // Transfer all RGB channels back to host
  cudaMemcpyAsync(hr_out, dr_out, num_pixels * sizeof(uchar), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(hg_out, dg_out, num_pixels * sizeof(uchar), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(hb_out, db_out, num_pixels * sizeof(uchar), cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);
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
