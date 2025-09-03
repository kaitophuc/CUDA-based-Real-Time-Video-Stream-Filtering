#ifndef BLUR_COMMON_HPP
#define BLUR_COMMON_HPP

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

// Common defines used across all kernels
#define BLUR_SIZE 128
#define TILE_DIM 32
#define DISTANCE 100
#define MAX_FACES 3

// Global managed memory for face detection data
extern __managed__ int *blur_x;
extern __managed__ int *blur_y;
extern __managed__ int *distance;
extern __managed__ int *num_faces;
extern bool enable;

// Kernel function pointer type
typedef void (*BlurKernelFunc)(cv::Mat&, int, int, int, int, uchar*, uchar*, uchar*, 
                              uchar*, uchar*, uchar*, uchar*, uchar*, uchar*, 
                              uchar*, uchar*, uchar*);

// Kernel performance structure
struct KernelPerformance {
    std::string name;
    BlurKernelFunc function;
    
    KernelPerformance(const std::string& n, BlurKernelFunc f) 
        : name(n), function(f) {}
};

// Common utility functions
__host__ void CheckCudaError(const std::string& error_message);
__host__ void AllocateHostMemory(uchar** h_buf, uchar** hr_in, uchar** hg_in, uchar** hb_in, 
                                uchar** hr_out, uchar** hg_out, uchar** hb_out, int num_pixels);
__host__ void AllocateDeviceMemory(uchar** d_buf, uchar** dr_in, uchar** dg_in, uchar** db_in, 
                                  uchar** dr_out, uchar** dg_out, uchar** db_out, int num_pixels);
__host__ void ReadImageFromFile(cv::Mat* image, uchar* hr_total, uchar* hg_total, uchar* hb_total,
                               int width, int height);

// Mouse callback and face detection functions
void OnMouse(int event, int x, int y, int, void* userdata);
cv::VideoCapture initializeVideoCapture(const std::string& video_file_path);
cv::dnn::Net initializeFaceDetection();
bool detectAndUpdateFace(cv::dnn::Net& net, cv::Mat& frame);

void testKernel(KernelPerformance& kernel, cv::VideoCapture& cap, cv::dnn::Net& net,
               int width, int height, int frames, int num_pixels,
               uchar* hr_in, uchar* hg_in, uchar* hb_in, 
               uchar* hr_out, uchar* hg_out, uchar* hb_out,
               uchar* dr_in, uchar* dg_in, uchar* db_in, 
               uchar* dr_out, uchar* dg_out, uchar* db_out);
void benchmarkKernel(KernelPerformance& kernel, cv::VideoCapture& cap, cv::dnn::Net& net,
                    int width, int height, int frames, int num_pixels,
                    uchar* hr_in, uchar* hg_in, uchar* hb_in, 
                    uchar* hr_out, uchar* hg_out, uchar* hb_out,
                    uchar* dr_in, uchar* dg_in, uchar* db_in, 
                    uchar* dr_out, uchar* dg_out, uchar* db_out,
                    double test_duration = 10.0);
void runInteractiveMode(KernelPerformance& kernel, cv::VideoCapture& cap, cv::dnn::Net& net,
                       int width, int height, int frames, int num_pixels,
                       uchar* hr_in, uchar* hg_in, uchar* hb_in, 
                       uchar* hr_out, uchar* hg_out, uchar* hb_out,
                       uchar* dr_in, uchar* dg_in, uchar* db_in, 
                       uchar* dr_out, uchar* dg_out, uchar* db_out);

#endif // BLUR_COMMON_HPP
