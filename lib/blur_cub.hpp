#ifndef BLUR_CUB_HPP
#define BLUR_CUB_HPP

#include "blur_common.hpp"

// CUB includes for optimized operations
#include <cub/cub.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/block/block_reduce.cuh>

const int BLOCK_THREADS = 128;
const int ITEMS_PER_THREAD = 15;

// CUB-optimized kernel declarations
template<int R, int BLOCK_THREADS = BLOCK_THREADS, int ITEMS_PER_THREAD = ITEMS_PER_THREAD>
__global__ void BoxBlurHorizontal(
    const uchar* d_in,
    int width, int height, int pitch,
    int *hsum);

template<int R, int BLOCK_THREADS = BLOCK_THREADS, int ITEMS_PER_THREAD = ITEMS_PER_THREAD>
__global__ void BoxBlurVertical(
    const int* hsum,
    const uchar* d_in,
    uchar* d_out,
    int width, int height, int pitch);

// CUB blur function declaration
void Blur_CUB(cv::Mat& frame, int width, int height, int frames, int num_pixels, uchar* hr_in, uchar* hg_in, uchar* hb_in, 
              uchar* hr_out, uchar* hg_out, uchar* hb_out, uchar* dr_in, uchar* dg_in, uchar* db_in, 
              uchar* dr_out, uchar* dg_out, uchar* db_out, cudaStream_t* streams);

#endif // BLUR_CUB_HPP
