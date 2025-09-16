#ifndef BLUR_THRUST_HPP
#define BLUR_THRUST_HPP

#include "blur_common.hpp"

// Thrust includes for parallel algorithms
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

// Thrust-optimized kernel declarations
template<int R>
__global__ void ThrustBoxBlurHorizontal(
    const uchar* d_in,
    int width, int height, int pitch,
    int *hsum);

template<int R>
__global__ void ThrustBoxBlurVertical(
    const int* hsum,
    const uchar* d_in,
    uchar* d_out,
    int width, int height, int pitch);

// Thrust blur function declaration
void Blur_Thrust(cv::Mat& frame, int width, int height, int frames, int num_pixels, uchar* hr_in, uchar* hg_in, uchar* hb_in, 
                uchar* hr_out, uchar* hg_out, uchar* hb_out, uchar* dr_in, uchar* dg_in, uchar* db_in, 
                uchar* dr_out, uchar* dg_out, uchar* db_out, cudaStream_t* streams);

#endif // BLUR_THRUST_HPP
