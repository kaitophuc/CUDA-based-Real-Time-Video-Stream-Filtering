#include "blur_common.hpp"
#include <iomanip>

// Global managed memory variables - defined here to be shared across modules
__managed__ int *blur_x;
__managed__ int *blur_y;
__managed__ int *distance;
__managed__ int *num_faces;
bool enable = true;

__host__ void CheckCudaError(const std::string& error_message) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    std::cerr << error_message << std::endl;
    exit(1);
  }
}

__host__ void AllocateHostMemory(uchar** h_buf, uchar** hr_in, uchar** hg_in, uchar** hb_in, uchar** hr_out, uchar** hg_out, 
                                 uchar** hb_out, int num_pixels) {
  cudaMallocHost(h_buf, 6 * num_pixels * sizeof(uchar));
  *hr_in = *h_buf;
  *hg_in = *hr_in + num_pixels;
  *hb_in = *hg_in + num_pixels;
  *hr_out = *hb_in + num_pixels;
  *hg_out = *hr_out + num_pixels;
  *hb_out = *hg_out + num_pixels;
  memset(*h_buf, 0, 6 * num_pixels * sizeof(uchar));
}

__host__ void AllocateDeviceMemory(uchar** d_buf, uchar** dr_in, uchar** dg_in, uchar** db_in, uchar** dr_out, uchar** dg_out, 
                                   uchar** db_out, int num_pixels) {
  size_t total_size = 6 * num_pixels * sizeof(uchar);
  cudaMalloc(d_buf, total_size);
  *dr_in = *d_buf;
  *dg_in = *dr_in + num_pixels;
  *db_in = *dg_in + num_pixels;
  *dr_out = *db_in + num_pixels;
  *dg_out = *dr_out + num_pixels;
  *db_out = *dg_out + num_pixels;
  cudaMemset(*d_buf, 0, total_size);
}

// Read the image from the file and store it in the host memory
__host__ void ReadImageFromFile(cv::Mat* image, uchar* hr_total, uchar* hg_total, uchar* hb_total,
                                int width, int height) {
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      cv::Vec3b pixel = image->at<cv::Vec3b>(i, j);
      hr_total[i * width + j] = pixel[2];
      hg_total[i * width + j] = pixel[1];
      hb_total[i * width + j] = pixel[0];
    }
  }
}

// Mouse callback function to get the coordinates of the mouse click
void OnMouse(int event, int x, int y, int, void* userdata) {
  if (event == cv::EVENT_LBUTTONDOWN) {
    // Add a new face position (up to MAX_FACES)
    if (*num_faces < MAX_FACES) {
      blur_x[*num_faces] = x;
      blur_y[*num_faces] = y;
      (*num_faces)++;
      enable = true;
    }
  } else if (event == cv::EVENT_RBUTTONDOWN) {
    // Clear all face positions
    *num_faces = 0;
    for (int i = 0; i < MAX_FACES; i++) {
      blur_x[i] = -1;
      blur_y[i] = -1;
    }
    enable = false;
  }
}

// Function to initialize video capture
cv::VideoCapture initializeVideoCapture(const std::string& video_file_path) {
  cv::VideoCapture cap;
  if (video_file_path == "0") {
    cap.open(0);
    // Set 1080p resolution if using webcam
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FPS, 60);
  } else {
    cap.open(video_file_path);
  }
  
  if (!cap.isOpened()) {
    std::cerr << "Error: Unable to open video file\n";
    exit(-1);
  }
  
  return cap;
}

// Function to initialize DNN model
cv::dnn::Net initializeFaceDetection() {
  std::string modelConfiguration = "./models/deploy.prototxt";
  std::string modelWeights = "./models/res10_300x300_ssd_iter_140000.caffemodel";
  cv::dnn::Net net = cv::dnn::readNetFromCaffe(modelConfiguration, modelWeights);
  
  try {
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    std::cout << "Using CUDA backend for face detection" << std::endl;
  } catch (const std::exception& e) {
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::cout << "Using CPU backend for face detection" << std::endl;
  }
  
  return net;
}

// Function to detect faces and update blur coordinates (selects top 3 faces by confidence)
bool detectAndUpdateFace(cv::dnn::Net& net, cv::Mat& frame) {
  if (!enable) {
    *num_faces = 0;
    for (int i = 0; i < MAX_FACES; i++) {
      blur_x[i] = -1;
      blur_y[i] = -1;
    }
    return false;
  }

  cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0));
  net.setInput(blob);
  cv::Mat detections = net.forward();

  const int numDetections = detections.size[2];
  const int numCoords = detections.size[3];
  float* data = (float*)detections.ptr<float>(0);

  // Store detected faces with confidence scores
  struct DetectedFace {
    int x, y, distance;
    float confidence;
  };
  std::vector<DetectedFace> detected_faces;

  for (int i = 0; i < numDetections; ++i) {
    float confidence = data[i * numCoords + 2];

    if (confidence > 0.5) {
      int x1 = static_cast<int>(data[i * numCoords + 3] * frame.cols);
      int y1 = static_cast<int>(data[i * numCoords + 4] * frame.rows);
      int x2 = static_cast<int>(data[i * numCoords + 5] * frame.cols);
      int y2 = static_cast<int>(data[i * numCoords + 6] * frame.rows);

      x1 = std::max(0, std::min(x1, frame.cols - 1));
      y1 = std::max(0, std::min(y1, frame.rows - 1));
      x2 = std::max(0, std::min(x2, frame.cols - 1));
      y2 = std::max(0, std::min(y2, frame.rows - 1));

      if (x2 > x1 && y2 > y1) {
        int center_x = (x1 + x2) / 2;
        int center_y = (y1 + y2) / 2;
        int face_distance = sqrt((x2 - center_x) * (x2 - center_x) + (y2 - center_y) * (y2 - center_y));
        
        detected_faces.push_back({center_x, center_y, face_distance, confidence});
        
        // Draw rectangle around detected face
        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
      }
    }
  }

  // Sort faces by confidence (highest first) and take up to MAX_FACES
  std::sort(detected_faces.begin(), detected_faces.end(), 
    [](const DetectedFace& a, const DetectedFace& b) {
      return a.confidence > b.confidence;
    });

  *num_faces = std::min(static_cast<int>(detected_faces.size()), MAX_FACES);
  
  for (int i = 0; i < *num_faces; i++) {
    blur_x[i] = detected_faces[i].x;
    blur_y[i] = detected_faces[i].y;
    distance[i] = detected_faces[i].distance;
  }
  
  // Clear remaining face slots
  for (int i = *num_faces; i < MAX_FACES; i++) {
    blur_x[i] = -1;
    blur_y[i] = -1;
  }

  return *num_faces > 0;
}

// Function to test a specific kernel
void testKernel(KernelPerformance& kernel, cv::VideoCapture& cap, cv::dnn::Net& net,
               int width, int height, int frames, int num_pixels,
               uchar* hr_in, uchar* hg_in, uchar* hb_in, 
               uchar* hr_out, uchar* hg_out, uchar* hb_out,
               uchar* dr_in, uchar* dg_in, uchar* db_in, 
               uchar* dr_out, uchar* dg_out, uchar* db_out) {
  
  std::cout << "\n=== Testing Kernel: " << kernel.name << " ===" << std::endl;
  std::cout << "Processing entire video..." << std::endl;
  
  cv::Mat frame;
  
  // Reset video to beginning for fair comparison
  cap.set(cv::CAP_PROP_POS_FRAMES, 0);

  double display_fps = 0.0;
  int frame_count = 0;
  auto start_time = std::chrono::high_resolution_clock::now();
  
  while (cap.read(frame)) {
    auto frame_start = std::chrono::high_resolution_clock::now();
    
    cv::flip(frame, frame, 1);
    ReadImageFromFile(&frame, hr_in, hg_in, hb_in, width, height);
    
    bool face_detected = detectAndUpdateFace(net, frame);
    
    if (face_detected) {
      // Apply the kernel
      kernel.function(frame, width, height, frames, num_pixels, 
                     hr_in, hg_in, hb_in, hr_out, hg_out, hb_out,
                     dr_in, dg_in, db_in, dr_out, dg_out, db_out);
    }

    auto frame_end = std::chrono::high_resolution_clock::now();
    auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start);
    double current_fps = 1000000.0 / frame_duration.count();

    display_fps += current_fps;
    frame_count++;

    // Add smoothed FPS and face count text (consistent with other modes)
    std::string fps_text = kernel.name + " - Realtime FPS: " + std::to_string(static_cast<int>(current_fps));
    std::string faces_text = "Faces detected: " + std::to_string(*num_faces) + "/3";
    cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    cv::putText(frame, faces_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
    cv::imshow("Kernel Testing", frame);
    cv::waitKey(1); // Allow OpenCV to update the display

  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  double final_display_fps = std::round((display_fps / frame_count) * 100.0) / 100.0;
  double final_total_fps = std::round((frame_count / (total_duration.count() / 1000.0)) * 100.0) / 100.0;

  std::cout << "Frames processed: " << frame_count << std::endl;
  std::cout << "Total time: " << total_duration.count() / 1000.0 << " seconds" << std::endl;
  std::cout << "Final display FPS: " << final_display_fps << std::endl;
  std::cout << "Final total FPS: " << final_total_fps << std::endl;
}

// Function to benchmark a specific kernel with webcam for a fixed duration
void benchmarkKernel(KernelPerformance& kernel, cv::VideoCapture& cap, cv::dnn::Net& net,
                    int width, int height, int frames, int num_pixels,
                    uchar* hr_in, uchar* hg_in, uchar* hb_in, 
                    uchar* hr_out, uchar* hg_out, uchar* hb_out,
                    uchar* dr_in, uchar* dg_in, uchar* db_in, 
                    uchar* dr_out, uchar* dg_out, uchar* db_out,
                    double test_duration) {
  
  std::cout << "\n=== Benchmarking Kernel: " << kernel.name << " (" << test_duration << " seconds) ===" << std::endl;
  std::cout << "Position yourself in front of the camera. Starting in 3 seconds..." << std::endl;
  
  // Warm-up period
  cv::Mat frame;
  for (int i = 0; i < 30; i++) {
    cap.read(frame);
    cv::flip(frame, frame, 1);
    std::string countdown_text = "Get ready... " + std::to_string(3 - i/10);
    cv::putText(frame, countdown_text, cv::Point(width/2 - 150, height/2), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 3);
    cv::imshow("Benchmark", frame);
    cv::waitKey(100);
  }
  
  std::cout << "Starting benchmark..." << std::endl;
  
  auto start_time = std::chrono::high_resolution_clock::now();
  double display_fps = 0.0;
  int frame_count = 0;
  
  while (true) {
    auto frame_start = std::chrono::high_resolution_clock::now();
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
    
    if (elapsed.count() / 1000.0 >= test_duration) {
      break;
    }
    
    if (!cap.read(frame)) {
      std::cerr << "Failed to read frame from webcam" << std::endl;
      break;
    }
    
    cv::flip(frame, frame, 1);
    ReadImageFromFile(&frame, hr_in, hg_in, hb_in, width, height);
    
    bool face_detected = detectAndUpdateFace(net, frame);
    
    if (face_detected) {
      // Apply the kernel
      kernel.function(frame, width, height, frames, num_pixels, 
                     hr_in, hg_in, hb_in, hr_out, hg_out, hb_out,
                     dr_in, dg_in, db_in, dr_out, dg_out, db_out);
    }
    
    auto frame_end = std::chrono::high_resolution_clock::now();
    auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start);
    double current_fps = 1000000.0 / frame_duration.count();

    display_fps += current_fps;
    frame_count++;
    
    // Display progress, current per-frame FPS, and face count
    double progress = (elapsed.count() / 1000.0) / test_duration;
    
    std::string progress_text = kernel.name + " - Progress: " + std::to_string(static_cast<int>(progress * 100)) + "%";
    std::string fps_text = "Current FPS: " + std::to_string(static_cast<int>(current_fps));
    std::string faces_text = "Faces detected: " + std::to_string(*num_faces) + "/3";
    std::string time_text = "Time: " + std::to_string(static_cast<int>(elapsed.count() / 1000.0)) + "/" + std::to_string(static_cast<int>(test_duration)) + "s";
    
    cv::putText(frame, progress_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    cv::putText(frame, fps_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(frame, faces_text, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
    cv::putText(frame, time_text, cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    // Progress bar (moved down to avoid overlap with face count text)
    int bar_width = 400;
    int bar_height = 20;
    cv::Point bar_start(width - bar_width - 20, 50);
    cv::Point bar_end(width - 20, bar_height + 50);
    cv::rectangle(frame, bar_start, bar_end, cv::Scalar(100, 100, 100), -1);
    cv::rectangle(frame, bar_start, cv::Point(bar_start.x + static_cast<int>(progress * bar_width), bar_end.y), cv::Scalar(0, 255, 0), -1);
    
    cv::imshow("Benchmark", frame);
    cv::waitKey(1);
  }
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  
  double final_display_fps = std::round((display_fps / frame_count) * 100.0) / 100.0;
  double final_total_fps = std::round((frame_count / (total_duration.count() / 1000.0)) * 100.0) / 100.0;
  
  std::cout << "Benchmark completed!" << std::endl;
  std::cout << "Frames processed: " << frame_count << std::endl;
  std::cout << "Total time: " << total_duration.count() / 1000.0 << " seconds" << std::endl;
  std::cout << "Average FPS: " << final_total_fps << std::endl;
  std::cout << "Final display FPS: " << final_display_fps << std::endl;
}

// Function to run interactive mode with selected kernel
void runInteractiveMode(KernelPerformance& kernel, cv::VideoCapture& cap, cv::dnn::Net& net,
                       int width, int height, int frames, int num_pixels,
                       uchar* hr_in, uchar* hg_in, uchar* hb_in, 
                       uchar* hr_out, uchar* hg_out, uchar* hb_out,
                       uchar* dr_in, uchar* dg_in, uchar* db_in, 
                       uchar* dr_out, uchar* dg_out, uchar* db_out) {
  
  std::cout << "\n=== Interactive Mode with " << kernel.name << " ===" << std::endl;
  std::cout << "Left click to enable blur (up to 3 faces), right click to disable, ESC to exit" << std::endl;
  
  cv::Mat frame;
  double display_fps = 0.0;
  int frame_count = 0;
  
  while (true) {
    auto frame_start = std::chrono::high_resolution_clock::now();
    
    if (!cap.read(frame)) {
      std::cout << "Video ended. Press ESC to exit." << std::endl;
      break; // Stop when video ends
    }

    cv::flip(frame, frame, 1);
    ReadImageFromFile(&frame, hr_in, hg_in, hb_in, width, height);

    bool face_detected = detectAndUpdateFace(net, frame);

    if (face_detected) {
      kernel.function(frame, width, height, frames, num_pixels, 
                     hr_in, hg_in, hb_in, hr_out, hg_out, hb_out,
                     dr_in, dg_in, db_in, dr_out, dg_out, db_out);
    }

    auto frame_end = std::chrono::high_resolution_clock::now();
    auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start);
    double current_fps = 1000000.0 / frame_duration.count();

    display_fps += current_fps;
    frame_count++;

    // Calculate running average for display
    double avg_display_fps = display_fps / frame_count;

    // Display average FPS and face count
    std::string fps_text = kernel.name + " - FPS: " + std::to_string(static_cast<int>(avg_display_fps));
    std::string faces_text = "Faces detected: " + std::to_string(*num_faces) + "/3";
    cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    cv::putText(frame, faces_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

    cv::imshow("Blurred Image", frame);
    cv::setMouseCallback("Blurred Image", OnMouse, &frame);
    
    int key = cv::waitKey(1);
    if (key == 27) break; // ESC key
  }
}
