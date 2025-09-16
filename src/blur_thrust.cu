#include "blur_thrust.hpp"
#include <thrust/device_ptr.h>

// Thrust functor for converting uchar to int
struct uchar_to_int {
    __host__ __device__
    int operator()(const uchar& x) const {
        return static_cast<int>(x);
    }
};

template<int R>
__global__ void ThrustBoxBlurHorizontal(
    const int* prefix_sums,
    int width, int height, int pitch,
    int *hsum)
{
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= height || col >= width) return;
    
    int left = max(0, col - R);
    int right = min(width - 1, col + R);
    
    // Use prefix sum for O(1) range query
    const int* row_prefix = prefix_sums + row * pitch;
    int sum = row_prefix[right + 1] - row_prefix[left];
    
    hsum[row * pitch + col] = sum;
}

template<int R>
__global__ void ThrustBoxBlurVertical(
    const int* col_prefix_sums,
    const uchar* d_in,
    uchar* d_out,
    int width, int height, int pitch)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col >= width || row >= height) return;
    
    int top = max(0, row - R);
    int bottom = min(height - 1, row + R);
    
    // Use column prefix sum for O(1) range query
    const int* column_prefix = col_prefix_sums + col * (height + 1);
    int sum = column_prefix[bottom + 1] - column_prefix[top];
    
    // Calculate pixel count
    int count = (min(width - 1, col + R) - max(0, col - R) + 1) * (bottom - top + 1);
    
    int avg = (count > 0) ? (sum / count) : int(d_in[row * pitch + col]);
    
    // Apply blur based on face detection
    int nfaces = min(*num_faces, MAX_FACES);
    bool should_blur = false;
    
    if (nfaces == 1) {
        int dx = col - blur_x[0];
        int dy = row - blur_y[0];
        should_blur = (dx * dx + dy * dy <= distance[0] * distance[0]);
    } else {
        #pragma unroll
        for (int face_idx = 0; face_idx < nfaces; face_idx++) {
            int dx = col - blur_x[face_idx];
            int dy = row - blur_y[face_idx];
            if (dx * dx + dy * dy <= distance[face_idx] * distance[face_idx]) {
                should_blur = true;
                break;
            }
        }
    }
    
    d_out[row * pitch + col] = should_blur ? static_cast<uchar>(avg) : d_in[row * pitch + col];
}


//==========================================================================//
// Thrust-based blur implementation
void Blur_Thrust(cv::Mat& frame, int width, int height, int frames, int num_pixels, 
                uchar* hr_in, uchar* hg_in, uchar* hb_in, 
                uchar* hr_out, uchar* hg_out, uchar* hb_out, 
                uchar* dr_in, uchar* dg_in, uchar* db_in, 
                uchar* dr_out, uchar* dg_out, uchar* db_out, cudaStream_t* streams) {

    // Create CUDA streams for parallel processing

    const int W = width;
    const int H = height;
    const int P = width;
    size_t bytesI = W * H * sizeof(int);
    size_t bytesP = (W + 1) * H * sizeof(int);

    // Allocate device memory for intermediate results
    int *d_hsum_r, *d_hsum_g, *d_hsum_b;
    int *d_row_prefix_r, *d_row_prefix_g, *d_row_prefix_b;
    int *d_col_prefix_r, *d_col_prefix_g, *d_col_prefix_b;
    
    cudaMalloc(&d_hsum_r, bytesI);
    cudaMalloc(&d_hsum_g, bytesI);
    cudaMalloc(&d_hsum_b, bytesI);
    cudaMalloc(&d_row_prefix_r, bytesP);
    cudaMalloc(&d_row_prefix_g, bytesP);
    cudaMalloc(&d_row_prefix_b, bytesP);
    cudaMalloc(&d_col_prefix_r, W * (H + 1) * sizeof(int));
    cudaMalloc(&d_col_prefix_g, W * (H + 1) * sizeof(int));
    cudaMalloc(&d_col_prefix_b, W * (H + 1) * sizeof(int));

    // Asynchronous memory transfers to device
    cudaMemcpyAsync(dr_in, hr_in, num_pixels * sizeof(uchar), cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(dg_in, hg_in, num_pixels * sizeof(uchar), cudaMemcpyHostToDevice, streams[1]);
    cudaMemcpyAsync(db_in, hb_in, num_pixels * sizeof(uchar), cudaMemcpyHostToDevice, streams[2]);

    // Process each channel using Thrust for prefix sum computation
    for (int channel = 0; channel < 3; channel++) {
        thrust::device_ptr<uchar> channel_in = thrust::device_pointer_cast((channel == 0) ? dr_in : (channel == 1) ? dg_in : db_in);
        thrust::device_ptr<int> row_prefix = thrust::device_pointer_cast((channel == 0) ? d_row_prefix_r : (channel == 1) ? d_row_prefix_g : d_row_prefix_b);
        thrust::device_ptr<int> col_prefix = thrust::device_pointer_cast((channel == 0) ? d_col_prefix_r : (channel == 1) ? d_col_prefix_g : d_col_prefix_b);
        thrust::device_ptr<int> hsum = thrust::device_pointer_cast((channel == 0) ? d_hsum_r : (channel == 1) ? d_hsum_g : d_hsum_b);

        // Compute row-wise prefix sums using Thrust
        for (int row = 0; row < H; row++) {
            thrust::device_ptr<uchar> row_input = channel_in + row * P;
            thrust::device_ptr<int> row_output = row_prefix + row * (W + 1);
            
            // Set first element to 0 for exclusive scan property
            cudaMemsetAsync(row_output.get(), 0, sizeof(int), streams[channel]);
            
            // Convert uchar to int and compute inclusive scan
            thrust::transform(thrust::cuda::par.on(streams[channel]), row_input, row_input + W, row_output + 1, uchar_to_int());
            thrust::inclusive_scan(thrust::cuda::par.on(streams[channel]), row_output + 1, row_output + W + 1, row_output + 1);
        }

        // Horizontal blur pass using row prefix sums
        dim3 h_grid((W + 255) / 256, H);
        dim3 h_block(256);
        ThrustBoxBlurHorizontal<BLUR_SIZE><<<h_grid, h_block, 0, streams[channel]>>>(
            row_prefix.get(), W, H, P, hsum.get());
        
        // Compute column-wise prefix sums using Thrust
        for (int col = 0; col < W; col++) {
             thrust::device_ptr<int> col_output = col_prefix + col * (H + 1);
             cudaMemsetAsync(col_output.get(), 0, sizeof(int), streams[channel]);

             // Create a temporary vector for the column
             thrust::device_vector<int> temp_col(H);
             for(int row=0; row < H; ++row){
                temp_col[row] = hsum[row*P + col];
             }
             
             // Copy to device memory and perform scan
             thrust::copy(thrust::cuda::par.on(streams[channel]), temp_col.begin(), temp_col.end(), col_output + 1);
             thrust::inclusive_scan(thrust::cuda::par.on(streams[channel]), col_output + 1, col_output + H + 1, col_output + 1);
        }
        
        // Vertical blur pass using column prefix sums
        thrust::device_ptr<uchar> channel_out = thrust::device_pointer_cast((channel == 0) ? dr_out : (channel == 1) ? dg_out : db_out);
        dim3 v_grid((W + 15) / 16, (H + 15) / 16);
        dim3 v_block(16, 16);
        ThrustBoxBlurVertical<BLUR_SIZE><<<v_grid, v_block, 0, streams[channel]>>>(
            col_prefix.get(), channel_in.get(), channel_out.get(), W, H, P);
    }

    // Asynchronous memory transfers back to host
    cudaMemcpyAsync(hr_out, dr_out, num_pixels * sizeof(uchar), cudaMemcpyDeviceToHost, streams[0]);
    cudaMemcpyAsync(hg_out, dg_out, num_pixels * sizeof(uchar), cudaMemcpyDeviceToHost, streams[1]);
    cudaMemcpyAsync(hb_out, db_out, num_pixels * sizeof(uchar), cudaMemcpyDeviceToHost, streams[2]);

    // Synchronize all streams
    for (int i = 0; i < 3; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // Free device memory
    cudaFree(d_hsum_r); cudaFree(d_hsum_g); cudaFree(d_hsum_b);
    cudaFree(d_row_prefix_r); cudaFree(d_row_prefix_g); cudaFree(d_row_prefix_b);
    cudaFree(d_col_prefix_r); cudaFree(d_col_prefix_g); cudaFree(d_col_prefix_b);

    // Update OpenCV frame with processed data
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cv::Vec3b& pixel = frame.at<cv::Vec3b>(i, j);
            pixel[2] = hr_out[i * width + j];  // Red
            pixel[1] = hg_out[i * width + j];  // Green  
            pixel[0] = hb_out[i * width + j];  // Blue
        }
    }
}

// Explicit template instantiation to avoid linking issues
template __global__ void ThrustBoxBlurHorizontal<BLUR_SIZE>(const int*, int, int, int, int*);
template __global__ void ThrustBoxBlurVertical<BLUR_SIZE>(const int*, const uchar*, uchar*, int, int, int);