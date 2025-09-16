#include "blur_common.hpp"
#include "blur_thrust.hpp"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

// Thrust functor for converting uchar to int
struct uchar_to_int {
    __host__ __device__
    int operator()(const uchar& x) const {
        return static_cast<int>(x);
    }
};

__global__ void BlurHorizontal(uchar* d_in, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int col_start = max(0, col - BLUR_SIZE);
    int col_end = min(width - 1, col + BLUR_SIZE);

    int global_idx = row * width + col;
    int global_start = row * width + col_start;
    int global_end = row * width + col_end;

    if (col < width && row < height) {
        if (col_start == 0) {
            d_in[global_idx] = d_in[global_end];
        } else {
            d_in[global_idx] = d_in[global_end] - d_in[global_start - 1];
        }
    }
}

__global__ void BlurVertical(uchar* d_in, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int row_start = max(0, row - BLUR_SIZE);
    int row_end = min(height - 1, row + BLUR_SIZE);

    int global_idx = row * width + col;
    int global_start = row_start * width + col;
    int global_end = row_end * width + col;

    if (col < width && row < height) {
        if (row_start == 0) {
            d_in[global_idx] = d_in[global_end];
        } else {
            d_in[global_idx] = d_in[global_end] - d_in[global_start - width];
        }
    }

    // Normalize the blurred value
    int count = box_count(col, row, width, height, BLUR_SIZE);
    d_in[global_idx] = static_cast<uchar>(d_in[global_idx] / count);
}   

void Blur_Thrust_Internal(cv::Mat& frame, int width, int height, int frames, int num_pixels, 
                uchar* hr_in, uchar* hg_in, uchar* hb_in, 
                uchar* hr_out, uchar* hg_out, uchar* hb_out, 
                cudaStream_t* streams) {
    const int W = width;
    const int H = height;

    thrust::device_vector<uchar> d_r_in(hr_in, hr_in + num_pixels);
    thrust::device_vector<uchar> d_g_in(hg_in, hg_in + num_pixels);
    thrust::device_vector<uchar> d_b_in(hb_in, hb_in + num_pixels);

    thrust::device_vector<uchar> d_r_out(num_pixels);
    thrust::device_vector<uchar> d_g_out(num_pixels);
    thrust::device_vector<uchar> d_b_out(num_pixels);

    for (int row = 0; row < H; ++row) {
        int row_start = row * W;
        int row_end = row_start + W;
        thrust::inclusive_scan(thrust::cuda::par.on(streams[0]), d_r_in.begin() + row_start, d_r_in.begin() + row_end, d_r_in.begin() + row_start);
        thrust::inclusive_scan(thrust::cuda::par.on(streams[1]), d_g_in.begin() + row_start, d_g_in.begin() + row_end, d_g_in.begin() + row_start);
        thrust::inclusive_scan(thrust::cuda::par.on(streams[2]), d_b_in.begin() + row_start, d_b_in.begin() + row_end, d_b_in.begin() + row_start);
    }

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((W + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (H + threadsPerBlock.y - 1) / threadsPerBlock.y); 
    BlurHorizontal<<<numBlocks, threadsPerBlock, 0, streams[0]>>>(
        d_r_out.data().get(), W, H);
    BlurHorizontal<<<numBlocks, threadsPerBlock, 0, streams[1]>>>(
        d_g_out.data().get(), W, H);
    BlurHorizontal<<<numBlocks, threadsPerBlock, 0, streams[2]>>>(
        d_b_out.data().get(), W, H);

    for (int i = 0; i < 3; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    for (int col = 0; col < W; ++col) {
        for (int row = 1; row < H; ++row) {
            int idx = row * W + col;
            d_r_out[idx] += d_r_out[(row - 1) * W + col];
            d_g_out[idx] += d_g_out[(row - 1) * W + col];
            d_b_out[idx] += d_b_out[(row - 1) * W + col];
        }
    }

    BlurVertical<<<numBlocks, threadsPerBlock, 0, streams[0]>>>(
        d_r_out.data().get(), W, H);
    BlurVertical<<<numBlocks, threadsPerBlock, 0, streams[1]>>>(
        d_g_out.data().get(), W, H);
    BlurVertical<<<numBlocks, threadsPerBlock, 0, streams[2]>>>(
        d_b_out.data().get(), W, H);

    thrust::copy(thrust::cuda::par.on(streams[0]), d_r_out.begin(), d_r_out.end(), hr_out);
    thrust::copy(thrust::cuda::par.on(streams[1]), d_g_out.begin(), d_g_out.end(), hg_out);
    thrust::copy(thrust::cuda::par.on(streams[2]), d_b_out.begin(), d_b_out.end(), hb_out);

    for (int i = 0; i < 3; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int nfaces = min(*num_faces, MAX_FACES);
            bool should_blur = false;
            if (nfaces == 1) {
                int dx = col - blur_x[0];
                int dy = row - blur_y[0];
                should_blur = (dx * dx + dy * dy <= distance[0] * distance[0]);
            }
            else {
                for (int f = 0; f < nfaces; f++) {
                    int dx = col - blur_x[f];
                    int dy = row - blur_y[f];
                    if (dx * dx + dy * dy <= distance[f] * distance[f]) {
                        should_blur = true;
                        break;
                    }
                }
            }
            cv::Vec3b& pixel = frame.at<cv::Vec3b>(row, col);
            pixel[2] = should_blur ? hr_out[row * width + col] : hr_in[row * width + col];
            pixel[1] = should_blur ? hg_out[row * width + col] : hg_in[row * width + col];
            pixel[0] = should_blur ? hb_out[row * width + col] : hb_in[row * width + col];
        }
    }

}

void Blur_Thrust(cv::Mat& frame, int width, int height, int frames, int num_pixels, 
                uchar* hr_in, uchar* hg_in, uchar* hb_in, 
                uchar* hr_out, uchar* hg_out, uchar* hb_out, 
                uchar* dr_in, uchar* dg_in, uchar* db_in, 
                uchar* dr_out, uchar* dg_out, uchar* db_out, cudaStream_t* streams) {

    Blur_Thrust_Internal(frame, width, height, frames, num_pixels, 
                hr_in, hg_in, hb_in, 
                hr_out, hg_out, hb_out,
                streams);
}