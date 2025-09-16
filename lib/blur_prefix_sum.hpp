/*
    Share same idea with blur_cub.hpp but using hand-crafted Brent-Kung prefix sum implementation for comparison.
*/

#ifndef BLUR_PREFIX_SUM_HPP
#define BLUR_PREFIX_SUM_HPP

#include "blur_common.hpp"
#include <cuda_runtime.h>

// Brent-Kung prefix sum device function
__device__ void Brent_Kung_Scan(int* data, int n);

// Box blur kernels using Brent-Kung prefix sum
template<int R, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BoxBlurHorizontal(
    const uchar* d_in,
    int width, int height, int pitch,
    int* hsum);

template<int R, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BoxBlurVertical(
    const int* hsum,
    const uchar* d_in,
    uchar* d_out,
    int width, int height, int pitch);

// Main blur function
void Blur_Brent_Kung(cv::Mat& frame, int width, int height, int frames, int num_pixels,
                     uchar* hr_in, uchar* hg_in, uchar* hb_in,
                     uchar* hr_out, uchar* hg_out, uchar* hb_out,
                     uchar* dr_in, uchar* dg_in, uchar* db_in,
                     uchar* dr_out, uchar* dg_out, uchar* db_out, cudaStream_t* streams);

#endif // BLUR_PREFIX_SUM_HPP