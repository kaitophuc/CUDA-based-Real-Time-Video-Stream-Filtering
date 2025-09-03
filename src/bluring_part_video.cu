// Main application file - orchestrates all blur kernels
#include "blur_common.hpp"
#include "blur_naive.hpp"
#include "blur_multistream.hpp"
#include "blur_cub.hpp"
#include "blur_prefix_sum.hpp"
#include <iomanip>
#include <thread>

int main(int argc, char** argv) {
  // Allocate Unified Memory for multiple faces
  cudaMallocManaged(&blur_x, MAX_FACES * sizeof(int));
  cudaMallocManaged(&blur_y, MAX_FACES * sizeof(int));
  cudaMallocManaged(&distance, MAX_FACES * sizeof(int));
  cudaMallocManaged(&num_faces, sizeof(int));

  // Initialize variables
  *num_faces = 0;
  for (int i = 0; i < MAX_FACES; i++) {
    blur_x[i] = -1;
    blur_y[i] = -1;
    distance[i] = DISTANCE;
  }

  // Read the video file path from the command line
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <video_file_path> <mode> [kernel]" << std::endl;
    std::cerr << "Modes: test, interactive" << std::endl;
    std::cerr << "Kernels (for interactive mode): naive, multistream, cub, brentkunng" << std::endl;
    std::cerr << "  test: Test all kernels and show performance comparison" << std::endl;
    std::cerr << "  interactive: Run specific kernel interactively" << std::endl;
    std::cerr << "Examples:" << std::endl;
    std::cerr << "  " << argv[0] << " 0 test                 # Test all kernels with webcam" << std::endl;
    std::cerr << "  " << argv[0] << " 0 interactive cub      # Run CUB kernel with webcam" << std::endl;
    std::cerr << "  " << argv[0] << " video.mp4 interactive naive # Run Naive kernel with video" << std::endl;
    return -1;
  }

  std::string video_file_path = argv[1];
  std::string mode = (argc >= 3) ? argv[2] : "interactive";
  std::string selected_kernel = (argc >= 4) ? argv[3] : "";

  // Validate mode
  if (mode != "test" && mode != "interactive") {
    std::cerr << "Error: Invalid mode '" << mode << "'. Valid modes: test, interactive" << std::endl;
    return -1;
  }

  // For interactive mode, kernel selection is required
  if (mode == "interactive" && selected_kernel.empty()) {
    std::cerr << "Error: Kernel selection required for interactive mode." << std::endl;
    std::cerr << "Valid kernels: naive, multistream, cub, brentkunng" << std::endl;
    return -1;
  }

  // Initialize components
  cv::dnn::Net net = initializeFaceDetection();
  cv::VideoCapture cap = initializeVideoCapture(video_file_path);

  int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  int frames = cap.get(cv::CAP_PROP_FPS);
  int num_pixels = width * height;

  std::cout << "Resolution: " << width << "x" << height << " @ " << frames << " FPS" << std::endl;

  // Allocate memory
  uchar *h_buf, *d_buf;
  uchar *dr_in, *dg_in, *db_in, *dr_out, *dg_out, *db_out;
  AllocateDeviceMemory(&d_buf, &dr_in, &dg_in, &db_in, &dr_out, &dg_out, &db_out, num_pixels);

  uchar* hr_in; uchar* hg_in; uchar* hb_in; uchar* hr_out; uchar* hg_out; uchar* hb_out;
  AllocateHostMemory(&h_buf, &hr_in, &hg_in, &hb_in, &hr_out, &hg_out, &hb_out, num_pixels);

  // Initialize available kernels - now using modular implementations
  std::vector<KernelPerformance> kernels = {
    KernelPerformance("Naive CUDA", Blur_Naive),
    KernelPerformance("Multi-Stream CUDA", Blur_MultiStream),
    KernelPerformance("CUB Optimized", Blur_CUB),
    KernelPerformance("Brent-Kung Prefix Sum", Blur_Brent_Kung)
  };

  try {
    if (mode == "test") {
      // Test all kernels
      std::cout << "\n=== KERNEL PERFORMANCE COMPARISON ===" << std::endl;
      
      for (auto& kernel : kernels) {
        testKernel(kernel, cap, net, width, height, frames, num_pixels,
                  hr_in, hg_in, hb_in, hr_out, hg_out, hb_out,
                  dr_in, dg_in, db_in, dr_out, dg_out, db_out);
      }
      
      // Print summary
      std::cout << "\n=== PERFORMANCE SUMMARY ===" << std::endl;
      std::cout << "All kernels have been tested. Check individual kernel outputs above for detailed performance metrics." << std::endl;
      
    } else if (mode == "interactive") {
      // Map kernel name to index
      int selected_index = -1;
      
      if (selected_kernel == "naive") {
        selected_index = 0;
      } else if (selected_kernel == "multistream") {
        selected_index = 1;
      } else if (selected_kernel == "cub") {
        selected_index = 2;
      } else if (selected_kernel == "brentkunng") {
        selected_index = 3;
      } else {
        std::cerr << "Error: Invalid kernel '" << selected_kernel << "'" << std::endl;
        std::cerr << "Valid kernels: naive, multistream, cub, brentkunng" << std::endl;
        return -1;
      }

      std::cout << "Running " << kernels[selected_index].name << " kernel in interactive mode..." << std::endl;
      
      runInteractiveMode(kernels[selected_index], cap, net, width, height, frames, num_pixels,
                        hr_in, hg_in, hb_in, hr_out, hg_out, hb_out,
                        dr_in, dg_in, db_in, dr_out, dg_out, db_out);
    }

    // Cleanup
    std::cout << "\nCleaning up resources..." << std::endl;
    
    // Close OpenCV windows first
    cv::destroyAllWindows();
    
    // Release video capture before DNN cleanup
    if (cap.isOpened()) {
      cap.release();
    }
    
    // Clear DNN network to prevent memory warnings
    net = cv::dnn::Net();
    
    // Small delay to ensure OpenCV cleanup completes
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Free unified memory first (these may be used by kernels)
    if (blur_x) {
      cudaFree(blur_x);
      blur_x = nullptr;
    }
    if (blur_y) {
      cudaFree(blur_y);
      blur_y = nullptr;
    }
    if (distance) {
      cudaFree(distance);
      distance = nullptr;
    }
    if (num_faces) {
      cudaFree(num_faces);
      num_faces = nullptr;
    }
    
    // Free CUDA device memory
    if (d_buf) {
      cudaFree(d_buf);
      d_buf = nullptr;
    }
    
    // Free CUDA host memory
    if (h_buf) {
      cudaFreeHost(h_buf);
      h_buf = nullptr;
    }
    
    // Synchronize before device reset
    cudaDeviceSynchronize();
    
    // Reset CUDA device
    cudaError_t resetError = cudaDeviceReset();
    if (resetError != cudaSuccess) {
      std::cerr << "Warning: CUDA device reset failed: " << cudaGetErrorString(resetError) << std::endl;
    }
    
    std::cout << "Cleanup completed." << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
