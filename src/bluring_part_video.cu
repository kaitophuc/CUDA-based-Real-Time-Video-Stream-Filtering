#include "../lib/bluring_part_video.hpp"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

int blur_x = -1;
int blur_y = -1;

__host__ void CheckCudaError(const std::string& error_message) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    std::cerr << error_message << std::endl;
    exit(1);
  }
}

__global__ void Convert(uchar* dr_in, uchar* dg_in, uchar* db_in, uchar* dr_out, uchar* dg_out, uchar* db_out, 
                        int idx, int width, int height, int x, int y) {
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int num_pixels = width * height;
    // Shared memory for the input image. Use for tiling the image to avoid bank conflicts.
    __shared__ uchar dr_in_shared[TILE_DIM][TILE_DIM];
    __shared__ uchar dg_in_shared[TILE_DIM][TILE_DIM];
    __shared__ uchar db_in_shared[TILE_DIM][TILE_DIM];
    if (col < width && row < height) {
      dr_in_shared[threadIdx.y][threadIdx.x] = dr_in[idx * num_pixels + row * width + col];
      dg_in_shared[threadIdx.y][threadIdx.x] = dg_in[idx * num_pixels + row * width + col];
      db_in_shared[threadIdx.y][threadIdx.x] = db_in[idx * num_pixels + row * width + col];
    } else {
      dr_in_shared[threadIdx.y][threadIdx.x] = 0;
      dg_in_shared[threadIdx.y][threadIdx.x] = 0;
      db_in_shared[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    if (col < width && row < height) {
      if ((col - x) * (col - x) + (row - y) * (row - y) <= DISTANCE * DISTANCE) {
        double pix_val_r = 0.00;
        double pix_val_g = 0.00;
        double pix_val_b = 0.00;
        for (int fRow = -FILTER_RADIUS; fRow <= FILTER_RADIUS; fRow++) {
            for (int fCol = -FILTER_RADIUS; fCol <= FILTER_RADIUS; fCol++) {
                int tileRow = threadIdx.y + fRow;
                int tileCol = threadIdx.x + fCol;
                if (tileRow >= 0 && tileRow < TILE_DIM && tileCol >= 0 && tileCol < TILE_DIM) {
                    pix_val_r += dr_in_shared[tileRow][tileCol] * Gaussian[fRow + FILTER_RADIUS][fCol + FILTER_RADIUS];
                    pix_val_g += dg_in_shared[tileRow][tileCol] * Gaussian[fRow + FILTER_RADIUS][fCol + FILTER_RADIUS];
                    pix_val_b += db_in_shared[tileRow][tileCol] * Gaussian[fRow + FILTER_RADIUS][fCol + FILTER_RADIUS];
                }
                else {
                    int imageRow = row + fRow;
                    int imageCol = col + fCol;
                    if (imageRow >= 0 && imageRow < height && imageCol >= 0 && imageCol < width) {
                        pix_val_r += dr_in[idx * num_pixels + imageRow * width + imageCol] * Gaussian[fRow + FILTER_RADIUS][fCol + FILTER_RADIUS];
                        pix_val_g += dg_in[idx * num_pixels + imageRow * width + imageCol] * Gaussian[fRow + FILTER_RADIUS][fCol + FILTER_RADIUS];
                        pix_val_b += db_in[idx * num_pixels + imageRow * width + imageCol] * Gaussian[fRow + FILTER_RADIUS][fCol + FILTER_RADIUS];
                    }
                }
            }
        }
        dr_out[idx * num_pixels + row * width + col] = static_cast<uchar>(pix_val_r);// - dr_in[idx * num_pixels + row * width + col]));
        dg_out[idx * num_pixels + row * width + col] = static_cast<uchar>(pix_val_g);// - dg_in[idx * num_pixels + row * width + col]));
        db_out[idx * num_pixels + row * width + col] = static_cast<uchar>(pix_val_b);// - db_in[idx * num_pixels + row * width + col]));
      } else {
        dr_out[idx * num_pixels + row * width + col] = dr_in[idx * num_pixels + row * width + col];
        dg_out[idx * num_pixels + row * width + col] = dg_in[idx * num_pixels + row * width + col];
        db_out[idx * num_pixels + row * width + col] = db_in[idx * num_pixels + row * width + col];

      }
    }
}

/*__global__ void Convert(uchar* dr_in, uchar* dg_in, uchar* db_in, uchar* dr_out, uchar* dg_out, uchar* db_out, 
                        int idx, int width, int height, int x, int y) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int num_pixels = width * height;
  // Shared memory for the input image. Use for tiling the image to avoid bank conflicts.
  __shared__ uchar dr_in_shared[TILE_DIM][TILE_DIM];
  __shared__ uchar dg_in_shared[TILE_DIM][TILE_DIM];
  __shared__ uchar db_in_shared[TILE_DIM][TILE_DIM];
  
  if (col < width && row < height) {
    dr_in_shared[threadIdx.y][threadIdx.x] = dr_in[idx * num_pixels + row * width + col];
    dg_in_shared[threadIdx.y][threadIdx.x] = dg_in[idx * num_pixels + row * width + col];
    db_in_shared[threadIdx.y][threadIdx.x] = db_in[idx * num_pixels + row * width + col];
  } else {
    dr_in_shared[threadIdx.y][threadIdx.x] = 0;
    dg_in_shared[threadIdx.y][threadIdx.x] = 0;
    db_in_shared[threadIdx.y][threadIdx.x] = 0;
  }

  __syncthreads();

  if (col < width && row < height) {
    if ((col - x) * (col - x) + (row - y) * (row - y) <= DISTANCE * DISTANCE) {
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
              pix_val_r += dr_in[idx * num_pixels + i * width + j];
              pix_val_g += dg_in[idx * num_pixels + i * width + j];
              pix_val_b += db_in[idx * num_pixels + i * width + j];
              ++pixels;
            }
          }
        }
      }

      dr_out[idx * num_pixels + row * width + col] = static_cast<uchar>(pix_val_r / pixels);
      dg_out[idx * num_pixels + row * width + col] = static_cast<uchar>(pix_val_g / pixels);
      db_out[idx * num_pixels + row * width + col] = static_cast<uchar>(pix_val_b / pixels);
    } else {
      dr_out[idx * num_pixels + row * width + col] = dr_in[idx * num_pixels + row * width + col];
      dg_out[idx * num_pixels + row * width + col] = dg_in[idx * num_pixels + row * width + col];
      db_out[idx * num_pixels + row * width + col] = db_in[idx * num_pixels + row * width + col];
    }
  }
}*/

// Copy the input image from the host to the device
__host__ void CopyFromHostToDevice(uchar* hr_in, uchar* hg_in, uchar* hb_in, uchar* dr_in, uchar* dg_in, uchar* db_in, 
                                   int count, int width, int height) {
  int num_pixels = count * width * height;
  size_t size = num_pixels * sizeof(uchar);
  cudaMemcpy(dr_in, hr_in, size, cudaMemcpyHostToDevice);
  CheckCudaError("Error copying from host to device");
  cudaMemcpy(dg_in, hg_in, size, cudaMemcpyHostToDevice);
  CheckCudaError("Error copying from host to device");
  cudaMemcpy(db_in, hb_in, size, cudaMemcpyHostToDevice);
  CheckCudaError("Error copying from host to device");
}

// Copy the result from the device to the host
__host__ void CopyFromDeviceToHost(uchar* dr_out, uchar* dg_out, uchar* db_out, uchar* hr_out, uchar* hg_out, 
                                   uchar* hb_out, int count, int width, int height) {
  int num_pixels = count * width * height;
  size_t size = num_pixels * sizeof(uchar);
  cudaMemcpy(hr_out, dr_out, size, cudaMemcpyDeviceToHost);
  CheckCudaError("Error copying from device to host");
  cudaMemcpy(hg_out, dg_out, size, cudaMemcpyDeviceToHost);
  CheckCudaError("Error copying from device to host");
  cudaMemcpy(hb_out, db_out, size, cudaMemcpyDeviceToHost);
  CheckCudaError("Error copying from device to host");
}

// Free the device memory
__host__ void FreeDeviceMemory(uchar* dr_in, uchar* dg_in, uchar* db_in, uchar* dr_out, uchar* dg_out, uchar* db_out) {
  cudaFree(dr_in);
  CheckCudaError("Error freeing device memory for dr_in");
  cudaFree(dg_in);
  CheckCudaError("Error freeing device memory for dg_in");
  cudaFree(db_in);
  CheckCudaError("Error freeing device memory for db_in");
  cudaFree(dr_out);
  CheckCudaError("Error freeing device memory for dr_out");
  cudaFree(dg_out);
  CheckCudaError("Error freeing device memory for dg_out");
  cudaFree(db_out);
  CheckCudaError("Error freeing device memory for db_out");
}

__host__ void CleanUp() {
  cudaDeviceReset();
  CheckCudaError("Error resetting device");
}

// Kernel to blur the image. Using dynamic parallelism to blur the image.
__global__ void Blur(uchar* dr_in, uchar* dg_in, uchar* db_in, uchar* dr_out, uchar* dg_out, uchar* db_out, 
                     int width, int height, int x, int y) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  dim3 block_size(TILE_DIM, TILE_DIM);
  dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
  Convert<<<grid_size, block_size>>>(dr_in, dg_in, db_in, dr_out, dg_out, db_out, idx, width, height, x, y);
}

__host__ void Execute(uchar* dr_in, uchar* dg_in, uchar* db_in, uchar* dr_out, uchar* dg_out, uchar* db_out, 
                      int count, int width, int height, int x, int y) {
  Blur<<<1, count>>>(dr_in, dg_in, db_in, dr_out, dg_out, db_out, width, height, x, y);
  CheckCudaError("Error executing kernel");
  cudaDeviceSynchronize();
}

// Read the image from the file and store it in the host memory
__host__ void ReadImageFromFile(cv::Mat* image, uchar* hr_total, uchar* hg_total, uchar* hb_total, int count, 
                                int width, int height) {
  int num_pixels = width * height;

  #pragma omp parallel for collapse(2)
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      cv::Vec3b pixel = image->at<cv::Vec3b>(i, j);
      hr_total[count * num_pixels + i * width + j] = pixel[2];
      hg_total[count * num_pixels + i * width + j] = pixel[1];
      hb_total[count * num_pixels + i * width + j] = pixel[0];
    }
  }
}

// Mouse callback function to get the coordinates of the mouse click
void OnMouse(int event, int x, int y, int, void* userdata) {
  cv::Mat* image = reinterpret_cast<cv::Mat*>(userdata);
  if (event == cv::EVENT_LBUTTONDOWN) {
    blur_x = x;
    blur_y = y;
  } else if (event == cv::EVENT_RBUTTONDOWN) {
    blur_x = -1;
    blur_y = -1;
  }
}

int main(int argc, char** argv) {
  // Read the video file path from the command line
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <video_file_path>" << std::endl;
    return -1;
  }

  std::string video_file_path = argv[1];

  try {
    // Create a window to display the blurred image
    cv::VideoCapture cap(video_file_path);
    if (!cap.isOpened()) {
      std::cerr << "Error: Unable to open video file\n";
      return -1;
    }

    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int frames = cap.get(cv::CAP_PROP_FPS);
    int num_pixels = width * height;

    std::cout << "Width: " << width << " Height: " << height << " Frames: " << frames << std::endl;

    uchar* dr_in;
    uchar* dg_in;
    uchar* db_in;
    uchar* dr_out;
    uchar* dg_out;
    uchar* db_out;

    // Allocate device memory
    cudaMalloc(&dr_in, num_pixels * NUM_FRAMES * sizeof(uchar));
    CheckCudaError("Error allocating device memory for dr_in");
    cudaMalloc(&dg_in, num_pixels * NUM_FRAMES * sizeof(uchar));
    CheckCudaError("Error allocating device memory for dg_in");
    cudaMalloc(&db_in, num_pixels * NUM_FRAMES * sizeof(uchar));
    CheckCudaError("Error allocating device memory for db_in");
    cudaMalloc(&dr_out, num_pixels * NUM_FRAMES * sizeof(uchar));
    CheckCudaError("Error allocating device memory for dr_out");
    cudaMalloc(&dg_out, num_pixels * NUM_FRAMES * sizeof(uchar));
    CheckCudaError("Error allocating device memory for dg_out");
    cudaMalloc(&db_out, num_pixels * NUM_FRAMES * sizeof(uchar));
    CheckCudaError("Error allocating device memory for db_out");

    // Allocate host memory
    uchar* hr_in = static_cast<uchar*>(malloc(num_pixels * NUM_FRAMES * sizeof(uchar)));
    uchar* hg_in = static_cast<uchar*>(malloc(num_pixels * NUM_FRAMES * sizeof(uchar)));
    uchar* hb_in = static_cast<uchar*>(malloc(num_pixels * NUM_FRAMES * sizeof(uchar)));

    uchar* hr_out = static_cast<uchar*>(malloc(num_pixels * NUM_FRAMES * sizeof(uchar)));
    uchar* hg_out = static_cast<uchar*>(malloc(num_pixels * NUM_FRAMES * sizeof(uchar)));
    uchar* hb_out = static_cast<uchar*>(malloc(num_pixels * NUM_FRAMES * sizeof(uchar)));

    cv::Mat frame;

    auto start = std::chrono::high_resolution_clock::now();

    while (true) {
      int count = 0;
      bool flag = true;
      // Read the video frames
      for (int i = 0; i < NUM_FRAMES; i++) {
        if (!cap.read(frame)) {
          flag = false;
          std::cerr << "Error: Unable to read video frame\n";
          auto end = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
          std::cout << "Time: " << duration.count() << " ms" << std::endl;
        }
        ReadImageFromFile(&frame, hr_in, hg_in, hb_in, count, width, height);
        ++count;
      }

      // Display the blurred image
      cv::setMouseCallback("Blurred Image", OnMouse, &frame);
      std::cout << blur_x << " " << blur_y << std::endl;

      // If no mouse click, display the original image
      if (blur_x == -1 && blur_y == -1) {
        cv::Mat output_image = cv::Mat::zeros(height, width, CV_8UC3);
        for (int idx = 0; idx < count; idx++) {
          #pragma omp parallel for collapse(2)
          for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
              cv::Vec3b pixel;
              pixel[2] = hr_in[idx * num_pixels + i * width + j];
              pixel[1] = hg_in[idx * num_pixels + i * width + j];
              pixel[0] = hb_in[idx * num_pixels + i * width + j];
              output_image.at<cv::Vec3b>(i, j) = pixel;
            }
          }
          cv::imshow("Blurred Image", output_image);
          if (cv::waitKey(1000 / frames) == 27) {
            flag = false;
            break;
          }
        }
      } else {
        CopyFromHostToDevice(hr_in, hg_in, hb_in, dr_in, dg_in, db_in, count, width, height);
        CheckCudaError("Error copying from host to device");
        Execute(dr_in, dg_in, db_in, dr_out, dg_out, db_out, count, width, height, blur_x, blur_y);
        CheckCudaError("Error executing kernel");
        CopyFromDeviceToHost(dr_out, dg_out, db_out, hr_out, hg_out, hb_out, count, width, height);
        CheckCudaError("Error copying from device to host");

        cv::Mat output_image = cv::Mat::zeros(height, width, CV_8UC3);
        for (int idx = 0; idx < count; idx++) {
          #pragma omp parallel for collapse(2)
          for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
              cv::Vec3b pixel;
              pixel[2] = hr_out[idx * num_pixels + i * width + j];
              pixel[1] = hg_out[idx * num_pixels + i * width + j];
              pixel[0] = hb_out[idx * num_pixels + i * width + j];
              output_image.at<cv::Vec3b>(i, j) = pixel;
            }
          }
          cv::imshow("Blurred Image", output_image);
          if (cv::waitKey(1000 / frames) == 27) {
            flag = false;
            break;
          }
        }
      }

      if (!flag) {
        break;
      }
    }

    // measure execution time
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
