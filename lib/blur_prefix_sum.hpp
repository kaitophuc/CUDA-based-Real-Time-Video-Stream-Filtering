/*
    Share same idea with blur_cub.hpp but using hand-crafted prefix sum implementation for comparison.
*/

#ifndef BLUR_PREFIX_SUM_HPP
#define BLUR_PREFIX_SUM_HPP

#include "blur_common.hpp"
#include <cuda_runtime.h>

// Brent-Kung prefix sum device functions
template<int BLOCK_SIZE>
__device__ void brent_kung_prefix_sum(int* data);

template<int THREADS_PER_BLOCK, int ITEMS_PER_THREAD>
__global__ void brent_kung_large_prefix_sum(
    const int* input, 
    int* output, 
    int* block_sums,
    int n);

__global__ void add_block_sums(int* data, const int* block_sums, int n, int elements_per_block);

// Box blur kernels using Brent-Kung prefix sum
template<int R, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BoxBlurHorizontal_BrentKung(
    const uchar* d_in,
    int width, int height, int pitch,
    int* hsum);

template<int R, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BoxBlurVertical_BrentKung(
    const int* hsum,
    const uchar* d_in,
    uchar* d_out,
    int width, int height, int pitch);

// Main blur function
void Blur_PrefixSum(cv::Mat& frame, int width, int height, int frames, int num_pixels,
                   uchar* hr_in, uchar* hg_in, uchar* hb_in,
                   uchar* hr_out, uchar* hg_out, uchar* hb_out,
                   uchar* dr_in, uchar* dg_in, uchar* db_in,
                   uchar* dr_out, uchar* dg_out, uchar* db_out);

#endif // BLUR_PREFIX_SUM_HPP