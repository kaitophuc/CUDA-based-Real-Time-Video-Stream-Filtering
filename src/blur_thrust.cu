#include "blur_common.hpp"
#include "blur_thrust.hpp"
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/functional.h>
struct uchar_to_int {
    __host__ __device__
    int operator()(const uchar& x) const {
        return static_cast<int>(x);
    }
};

struct horizontal_blur_functor {
    const int* prefix_sum_ptr;
    const int width;

    horizontal_blur_functor(const int* ptr, int w)
        : prefix_sum_ptr(ptr), width(w) {}

    __host__ __device__
    int operator()(const int& i) const {
        int row = i / width;
        int row_start_idx = row * width;

        int r = i + BLUR_SIZE;
        int l_minus_1 = i - BLUR_SIZE - 1;

        // Clamp 'r' and 'l_minus_1' to the boundaries of the current row
        r = min(r, row_start_idx + width - 1);
        l_minus_1 = max(l_minus_1, row_start_idx - 1);

        int right_val = prefix_sum_ptr[r];
        int left_val = (l_minus_1 < row_start_idx) ? 0 : prefix_sum_ptr[l_minus_1];

        return right_val - left_val;
    }
};

struct vertical_blur_functor {
    const int* prefix_sum_ptr;
    const uchar* original_in_ptr;
    const int width;
    const int height;
    const int* blur_x_ptr;
    const int* blur_y_ptr;
    const int* distance_ptr;
    const int* num_faces_ptr;

    vertical_blur_functor(const int* p_ptr, const uchar* o_ptr, int w, int h, int* bx, int* by, int* d, int* nf)
        : prefix_sum_ptr(p_ptr), original_in_ptr(o_ptr), width(w), height(h),
          blur_x_ptr(bx), blur_y_ptr(by), distance_ptr(d), num_faces_ptr(nf) {}

    __host__ __device__
    uchar operator()(const int& i) const {
        int col = i % width;
        int row = i / width;

        int r = row + BLUR_SIZE;
        int l_minus_1 = row - BLUR_SIZE - 1;

        r = min(r, height - 1);

        int right_val = prefix_sum_ptr[r * width + col];
        int left_val = (l_minus_1 < 0) ? 0 : prefix_sum_ptr[l_minus_1 * width + col];

        int sum = right_val - left_val;
        int count = box_count(col, row, width, height, BLUR_SIZE);
        uchar blurred_val = (count > 0) ? static_cast<uchar>(sum / count) : original_in_ptr[i];

        // Check if the pixel should be blurred based on face locations
        bool should_blur = false;
        int nfaces = min(*num_faces_ptr, MAX_FACES);

        for (int f = 0; f < nfaces; f++) {
            int dx = col - blur_x[f];
            int dy = row - blur_y[f];
            if (dx * dx + dy * dy <= distance[f] * distance[f]) {
                should_blur = true;
                break;
            }
        }

        return should_blur ? blurred_val : original_in_ptr[i];
    }
};

void Blur_Thrust_Channel(const thrust::device_vector<uchar>& d_in,
                         thrust::device_vector<uchar>& d_out,
                         int width, int height, cudaStream_t stream) {
    const int num_pixels = width * height;

    thrust::device_vector<int> d_int_in(num_pixels);
    thrust::transform(thrust::cuda::par.on(stream), d_in.begin(), d_in.end(), d_int_in.begin(), uchar_to_int());

    thrust::device_vector<int> d_row_prefix_sum(num_pixels);

    for (int row = 0; row < height; ++row) {
        int row_start = row * width;
        thrust::inclusive_scan(thrust::cuda::par.on(stream),
                               d_int_in.begin() + row_start,
                               d_int_in.begin() + row_start + width,
                               d_row_prefix_sum.begin() + row_start);
    }

    thrust::device_vector<int> d_horizontal_sum(num_pixels);
    thrust::transform(thrust::cuda::par.on(stream),
                      thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(num_pixels),
                      d_horizontal_sum.begin(),
                      horizontal_blur_functor(thrust::raw_pointer_cast(d_row_prefix_sum.data()), width));

    thrust::device_vector<int> d_col_prefix_sum = d_horizontal_sum;

    for (int col = 0; col < width; ++col) {
        auto col_begin = thrust::make_permutation_iterator(
            d_col_prefix_sum.begin(),
            thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                            [=] __device__ (int i) { return i * width + col; })
        );
        // Perform the scan out-of-place by reading from d_horizontal_sum and writing to d_col_prefix_sum
        thrust::inclusive_scan(thrust::cuda::par.on(stream),
                               col_begin,
                               col_begin + height,
                               col_begin);
    }

    thrust::transform(thrust::cuda::par.on(stream),
                      thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(num_pixels),
                      d_out.begin(),
                      vertical_blur_functor(thrust::raw_pointer_cast(d_col_prefix_sum.data()),
                                           thrust::raw_pointer_cast(d_in.data()),
                                           width, height, blur_x, blur_y, distance, num_faces));
}


void Blur_Thrust(cv::Mat& frame, int width, int height, int frames, int num_pixels,
                uchar* hr_in, uchar* hg_in, uchar* hb_in,
                uchar* hr_out, uchar* hg_out, uchar* hb_out,
                uchar* dr_in_ptr, uchar* dg_in_ptr, uchar* db_in_ptr,
                uchar* dr_out_ptr, uchar* dg_out_ptr, uchar* db_out_ptr, cudaStream_t* streams) {


    thrust::device_ptr<uchar> dr_in = thrust::device_pointer_cast(dr_in_ptr);
    thrust::device_ptr<uchar> dg_in = thrust::device_pointer_cast(dg_in_ptr);
    thrust::device_ptr<uchar> db_in = thrust::device_pointer_cast(db_in_ptr);
    thrust::device_ptr<uchar> dr_out = thrust::device_pointer_cast(dr_out_ptr);
    thrust::device_ptr<uchar> dg_out = thrust::device_pointer_cast(dg_out_ptr);
    thrust::device_ptr<uchar> db_out = thrust::device_pointer_cast(db_out_ptr);

    cudaMemcpyAsync(dr_in_ptr, hr_in, num_pixels * sizeof(uchar), cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(dg_in_ptr, hg_in, num_pixels * sizeof(uchar), cudaMemcpyHostToDevice, streams[1]);
    cudaMemcpyAsync(db_in_ptr, hb_in, num_pixels * sizeof(uchar), cudaMemcpyHostToDevice, streams[2]);

    // Wait for memory transfers to complete before creating device vectors
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaStreamSynchronize(streams[2]);

    // Create device vectors by copying from the device memory
    thrust::device_vector<uchar> d_r_in(num_pixels);
    thrust::device_vector<uchar> d_g_in(num_pixels);
    thrust::device_vector<uchar> d_b_in(num_pixels);
    thrust::device_vector<uchar> d_r_out(num_pixels);
    thrust::device_vector<uchar> d_g_out(num_pixels);
    thrust::device_vector<uchar> d_b_out(num_pixels);

    // Copy data from device pointers to device vectors
    thrust::copy(dr_in, dr_in + num_pixels, d_r_in.begin());
    thrust::copy(dg_in, dg_in + num_pixels, d_g_in.begin());
    thrust::copy(db_in, db_in + num_pixels, d_b_in.begin());

    Blur_Thrust_Channel(d_r_in, d_r_out, width, height, streams[0]);
    Blur_Thrust_Channel(d_g_in, d_g_out, width, height, streams[1]);
    Blur_Thrust_Channel(d_b_in, d_b_out, width, height, streams[2]);

    // Copy results back from device vectors to device pointers
    thrust::copy(d_r_out.begin(), d_r_out.end(), dr_out);
    thrust::copy(d_g_out.begin(), d_g_out.end(), dg_out);
    thrust::copy(d_b_out.begin(), d_b_out.end(), db_out);

    cudaMemcpyAsync(hr_out, dr_out_ptr, num_pixels * sizeof(uchar), cudaMemcpyDeviceToHost, streams[0]);
    cudaMemcpyAsync(hg_out, dg_out_ptr, num_pixels * sizeof(uchar), cudaMemcpyDeviceToHost, streams[1]);
    cudaMemcpyAsync(hb_out, db_out_ptr, num_pixels * sizeof(uchar), cudaMemcpyDeviceToHost, streams[2]);

    for (int i = 0; i < 3; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    #pragma omp parallel for collapse(2)
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            cv::Vec3b& pixel = frame.at<cv::Vec3b>(row, col);
            pixel[2] = hr_out[row * width + col];
            pixel[1] = hg_out[row * width + col];
            pixel[0] = hb_out[row * width + col];
        }
    }
}