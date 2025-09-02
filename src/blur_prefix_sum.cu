#include "blur_prefix_sum.hpp"
// Box count function to calculate the number of pixels in the blur kernel
__device__ __forceinline__ int box_count(int x, int y, int width, int height, int R) {
    int left = max(0, x - R);
    int right = min(width - 1, x + R);
    int top = max(0, y - R);
    int bottom = min(height - 1, y + R);
    return (right - left + 1) * (bottom - top + 1);
}

// Optimized Brent-Kung algorithm for prefix sum within a block
template<int BLOCK_SIZE>
__device__ void brent_kung_prefix_sum(int* data) {
    const int tid = threadIdx.x;
    const int n = BLOCK_SIZE;
    
    // Up-sweep phase (reduce) - builds binary tree
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = (tid + 1) * (n / d) - 1;
            int bi = (tid + 1) * (n / d) + (n / d) / 2 - 1;
            if (bi < n) {
                data[bi] += data[ai];
            }
        }
    }
    
    // Clear the last element (root becomes identity)
    if (tid == 0) {
        data[n - 1] = 0;
    }
    
    // Down-sweep phase (distribute) - traverses tree down
    for (int d = 1; d < n; d <<= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = (tid + 1) * (n / d) - 1;
            int bi = (tid + 1) * (n / d) + (n / d) / 2 - 1;
            if (bi < n) {
                int temp = data[ai];
                data[ai] = data[bi];
                data[bi] += temp;
            }
        }
    }
    __syncthreads();
}

// Multi-block Brent-Kung prefix sum for large arrays
template<int THREADS_PER_BLOCK, int ITEMS_PER_THREAD>
__global__ void brent_kung_large_prefix_sum(
    const int* input, 
    int* output, 
    int* block_sums,
    int n) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_start = bid * THREADS_PER_BLOCK * ITEMS_PER_THREAD;
    
    __shared__ int shared_data[THREADS_PER_BLOCK * ITEMS_PER_THREAD];
    
    // Load data into shared memory
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = block_start + tid * ITEMS_PER_THREAD + i;
        int shared_idx = tid * ITEMS_PER_THREAD + i;
        
        if (idx < n) {
            shared_data[shared_idx] = input[idx];
        } else {
            shared_data[shared_idx] = 0;
        }
    }
    __syncthreads();
    
    // Apply Brent-Kung within block
    brent_kung_prefix_sum<THREADS_PER_BLOCK * ITEMS_PER_THREAD>(shared_data);
    
    // Store block sum for later propagation
    if (tid == 0 && block_sums != nullptr) {
        int last_valid = min(THREADS_PER_BLOCK * ITEMS_PER_THREAD - 1, n - block_start - 1);
        if (last_valid >= 0) {
            block_sums[bid] = shared_data[last_valid] + 
                              (block_start + last_valid < n ? input[block_start + last_valid] : 0);
        }
    }
    
    // Write results back to global memory
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = block_start + tid * ITEMS_PER_THREAD + i;
        int shared_idx = tid * ITEMS_PER_THREAD + i;
        
        if (idx < n) {
            output[idx] = shared_data[shared_idx];
        }
    }
}

// Kernel to add block sums to all elements
__global__ void add_block_sums(int* data, const int* block_sums, int n, int elements_per_block) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && blockIdx.x > 0) {
        data[idx] += block_sums[blockIdx.x - 1];
    }
}

template<int R, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BoxBlurHorizontal_BrentKung(
    const uchar* d_in,
    int width, int height, int pitch,
    int* hsum) {
    
    constexpr int TILE = BLOCK_THREADS * ITEMS_PER_THREAD;
    
    int row = blockIdx.y;
    if (row >= height) return;
    
    int x0 = blockIdx.x * TILE;
    int valid = max(0, min(TILE, width - x0));
    
    __shared__ int shared_data[TILE];  // +1 for prefix sum sentinel
    
    const int tid = threadIdx.x;
    
    // Load pixel data into shared memory
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int x_local = tid * ITEMS_PER_THREAD + i;
        int x_global = x0 + x_local;
        
        if (x_local < valid && x_global < width) {
            shared_data[x_local] = static_cast<int>(d_in[row * pitch + x_global]);
        } else {
            shared_data[x_local] = 0;
        }
    }
    
    __syncthreads();
    
    // Store original values before prefix sum computation
    __shared__ int original_data[TILE];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int x_local = tid * ITEMS_PER_THREAD + i;
        if (x_local < valid) {
            original_data[x_local] = shared_data[x_local];
        }
    }
    __syncthreads();
    
    // Apply Brent-Kung prefix sum (converts to exclusive prefix sum)
    brent_kung_prefix_sum<TILE>(shared_data);
    
    // Calculate box blur sums for each pixel using optimized prefix sum
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int x_local = tid * ITEMS_PER_THREAD + i;
        int x_global = x0 + x_local;
        
        if (x_local < valid && x_global < width) {
            int left = max(0, x_global - R);
            int right = min(width - 1, x_global + R);
            
            int sum = 0;
            
            // Use original values for pixels within current tile
            int L = max(0, left - x0);
            int R_local = min(valid - 1, right - x0);
            
            if (L <= R_local) {
                // Use exclusive prefix sum: sum[L..R_local] = (E[R_local] + a[R_local]) - E[L]
                int E_R = shared_data[R_local];
                int E_L = (L >= 0) ? shared_data[L] : 0; // L is >=0 by construction
                sum += (E_R + original_data[R_local]) - E_L;
            }
            
            // Add pixels outside current tile (boundary handling)
            for (int c = left; c < x0; c++) {
                if (c >= 0) {
                    sum += static_cast<int>(d_in[row * pitch + c]);
                }
            }
            
            for (int c = x0 + valid; c <= right; c++) {
                if (c < width) {
                    sum += static_cast<int>(d_in[row * pitch + c]);
                }
            }
            
            hsum[row * pitch + x_global] = sum;
        }
    }
}

template<int R, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BoxBlurVertical_BrentKung(
    const int* hsum,
    const uchar* d_in,
    uchar* d_out,
    int width, int height, int pitch) {
    
    constexpr int TILE = BLOCK_THREADS * ITEMS_PER_THREAD;
    
    int col = blockIdx.x;
    int y0 = blockIdx.y * TILE;
    int valid = max(0, min(TILE, height - y0));
    
    if (col >= width) return;
    
    __shared__ int shared_sums[TILE];  
    
    const int tid = threadIdx.x;
    int nfaces = min(*num_faces, MAX_FACES);
    
    // Load horizontal sums into shared memory
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int y_local = tid * ITEMS_PER_THREAD + i;
        int y_global = y0 + y_local;
        
        if (y_local < valid && y_global < height) {
            shared_sums[y_local] = hsum[y_global * pitch + col];
        } else {
            shared_sums[y_local] = 0;
        }
    }
    
    __syncthreads();
    
    // Store original values before prefix sum computation
    __shared__ int original_sums[TILE];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int y_local = tid * ITEMS_PER_THREAD + i;
        if (y_local < valid) {
            original_sums[y_local] = shared_sums[y_local];
        }
    }
    __syncthreads();
    
    // Apply Brent-Kung prefix sum for vertical direction
    brent_kung_prefix_sum<TILE>(shared_sums);
    
    // Calculate final blur values
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int y_local = tid * ITEMS_PER_THREAD + i;
        int y_global = y0 + y_local;
        
        if (y_local < valid && y_global < height) {
            int top = max(0, y_global - R);
            int bottom = min(height - 1, y_global + R);
            
            int sum = 0;
            
            // Use prefix sum for pixels within current tile (O(1) operation!)
            int T = max(0, top - y0);
            int B = min(valid - 1, bottom - y0);
            
            if (T <= B) {
                // Use exclusive prefix sum: sum[T..B] = (E[B] + a[B]) - E[T]
                int E_B = shared_sums[B];
                int E_T = (T >= 0) ? shared_sums[T] : 0; // T is >=0 by construction
                sum += (E_B + original_sums[B]) - E_T;
            }
            
            // Add pixels outside current tile (boundary handling)
            for (int r = top; r < y0; r++) {
                if (r >= 0) {
                    sum += hsum[r * pitch + col];
                }
            }
            
            for (int r = y0 + valid; r <= bottom; r++) {
                if (r < height) {
                    sum += hsum[r * pitch + col];
                }
            }
            
            // Calculate average using the box_count function (like CUB implementation)
            int count = box_count(col, y_global, width, height, R);
            int avg = (count > 0) ? (sum / count) : static_cast<int>(d_in[y_global * pitch + col]);
            
            // Face detection logic
            bool should_blur = false;
            if (nfaces == 1) {
                int dx = col - blur_x[0];
                int dy = y_global - blur_y[0];
                should_blur = (dx * dx + dy * dy <= distance[0] * distance[0]);
            } else {
                #pragma unroll
                for (int face_idx = 0; face_idx < nfaces; face_idx++) {
                    int dx = col - blur_x[face_idx];
                    int dy = y_global - blur_y[face_idx];
                    if (dx * dx + dy * dy <= distance[face_idx] * distance[face_idx]) {
                        should_blur = true;
                        break;
                    }
                }
            }
            
            // Output final result
            d_out[y_global * pitch + col] = should_blur ? 
                static_cast<uchar>(max(0, min(255, avg))) : 
                d_in[y_global * pitch + col];
        }
    }
}

void Blur_PrefixSum(cv::Mat& frame, int width, int height, int frames, int num_pixels,
                   uchar* hr_in, uchar* hg_in, uchar* hb_in,
                   uchar* hr_out, uchar* hg_out, uchar* hb_out,
                   uchar* dr_in, uchar* dg_in, uchar* db_in,
                   uchar* dr_out, uchar* dg_out, uchar* db_out) {
    
    const int W = width;
    const int H = height;
    const int P = width;  
    
    // Allocate memory for horizontal sums
    int *d_hsum_r, *d_hsum_g, *d_hsum_b;
    size_t bytesI = W * H * sizeof(int);
    
    cudaMalloc(&d_hsum_r, bytesI);
    cudaMalloc(&d_hsum_g, bytesI);
    cudaMalloc(&d_hsum_b, bytesI);
    
    // Create streams for parallel processing
    cudaStream_t streams[3];
    for (int i = 0; i < 3; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Grid and block configurations
    constexpr int BLOCK_THREADS = 128;
    constexpr int ITEMS_PER_THREAD = 8;
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    
    dim3 bh(BLOCK_THREADS);
    dim3 gh((W + TILE_SIZE - 1) / TILE_SIZE, H);
    dim3 gv(W, (H + TILE_SIZE - 1) / TILE_SIZE);
    
    // Memory transfers H2D
    cudaMemcpyAsync(dr_in, hr_in, W * H * sizeof(uchar), cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(dg_in, hg_in, W * H * sizeof(uchar), cudaMemcpyHostToDevice, streams[1]);
    cudaMemcpyAsync(db_in, hb_in, W * H * sizeof(uchar), cudaMemcpyHostToDevice, streams[2]);
    
    // Horizontal blur pass
    BoxBlurHorizontal_BrentKung<BLUR_SIZE, BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<gh, bh, 0, streams[0]>>>(dr_in, W, H, P, d_hsum_r);
    BoxBlurHorizontal_BrentKung<BLUR_SIZE, BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<gh, bh, 0, streams[1]>>>(dg_in, W, H, P, d_hsum_g);
    BoxBlurHorizontal_BrentKung<BLUR_SIZE, BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<gh, bh, 0, streams[2]>>>(db_in, W, H, P, d_hsum_b);
    
    // Vertical blur pass with face detection
    BoxBlurVertical_BrentKung<BLUR_SIZE, BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<gv, bh, 0, streams[0]>>>(d_hsum_r, dr_in, dr_out, W, H, P);
    BoxBlurVertical_BrentKung<BLUR_SIZE, BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<gv, bh, 0, streams[1]>>>(d_hsum_g, dg_in, dg_out, W, H, P);
    BoxBlurVertical_BrentKung<BLUR_SIZE, BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<gv, bh, 0, streams[2]>>>(d_hsum_b, db_in, db_out, W, H, P);
    
    // Memory transfers D2H
    cudaMemcpyAsync(hr_out, dr_out, W * H * sizeof(uchar), cudaMemcpyDeviceToHost, streams[0]);
    cudaMemcpyAsync(hg_out, dg_out, W * H * sizeof(uchar), cudaMemcpyDeviceToHost, streams[1]);
    cudaMemcpyAsync(hb_out, db_out, W * H * sizeof(uchar), cudaMemcpyDeviceToHost, streams[2]);
    
    // Synchronize streams
    for (int i = 0; i < 3; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    // Copy blurred data back to OpenCV frame
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cv::Vec3b& pixel = frame.at<cv::Vec3b>(i, j);
            pixel[0] = hb_out[i * width + j];  // Blue
            pixel[1] = hg_out[i * width + j];  // Green
            pixel[2] = hr_out[i * width + j];  // Red
        }
    }
    
    // Cleanup
    cudaFree(d_hsum_r);
    cudaFree(d_hsum_g);
    cudaFree(d_hsum_b);
    
    CheckCudaError("Blur_PrefixSum execution failed");
}