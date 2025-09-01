#ifndef BLUR_MULTISTREAM_HPP
#define BLUR_MULTISTREAM_HPP

#include "blur_common.hpp"

// Multi-stream CUDA kernel declaration
__global__ void Convert_MultiStream(uchar* d_in, uchar* d_out, int width, int height);

// Multi-stream blur function declaration
void Blur_MultiStream(cv::Mat& frame, int width, int height, int frames, int num_pixels, uchar* hr_in, uchar* hg_in, uchar* hb_in, 
                   uchar* hr_out, uchar* hg_out, uchar* hb_out, uchar* dr_in, uchar* dg_in, uchar* db_in, 
                   uchar* dr_out, uchar* dg_out, uchar* db_out);

#endif // BLUR_MULTISTREAM_HPP
