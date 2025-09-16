#ifndef BLUR_NAIVE_HPP
#define BLUR_NAIVE_HPP

#include "blur_common.hpp"

// Naive CUDA kernel declaration
__global__ void Convert_Naive(uchar* dr_in, uchar* dg_in, uchar* db_in, uchar* dr_out, uchar* dg_out, uchar* db_out, 
                        int width, int height);

// Naive blur function declaration
void Blur_Naive(cv::Mat& frame, int width, int height, int frames, int num_pixels, uchar* hr_in, uchar* hg_in, uchar* hb_in, 
           uchar* hr_out, uchar* hg_out, uchar* hb_out, uchar* dr_in, uchar* dg_in, uchar* db_in, 
           uchar* dr_out, uchar* dg_out, uchar* db_out, cudaStream_t* streams);

#endif // BLUR_NAIVE_HPP
