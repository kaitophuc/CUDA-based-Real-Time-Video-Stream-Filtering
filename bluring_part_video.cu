#include "bluring_part_video.hpp"

using namespace std;
/*#######################################################*/
int blur_x = -1, blur_y = -1;

__host__ void check(string error)
{
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        cout << "Error: " << cudaGetErrorString(err) << endl;
        cout << error << endl;
        exit(1);
    }
}
__global__ void convert (uchar *dr_in, uchar *dg_in, uchar *db_in, uchar *dr_out, uchar *dg_out, uchar *db_out, int idx, int width, int height, int x, int y) {
    //printf("%d\n", idx);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int numPixels = width * height;
    __shared__ uchar dr_in_shared [TILE_DIM][TILE_DIM];
    __shared__ uchar dg_in_shared [TILE_DIM][TILE_DIM];
    __shared__ uchar db_in_shared [TILE_DIM][TILE_DIM];
    if (col < width && row < height) {
        dr_in_shared [threadIdx.y][threadIdx.x] = dr_in[idx * numPixels + row * width + col];
        dg_in_shared [threadIdx.y][threadIdx.x] = dg_in[idx * numPixels + row * width + col];
        db_in_shared [threadIdx.y][threadIdx.x] = db_in[idx * numPixels + row * width + col];
    } else {
        dr_in_shared [threadIdx.y][threadIdx.x] = 0;
        dg_in_shared [threadIdx.y][threadIdx.x] = 0;
        db_in_shared [threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads ();

    if (col < width && row < height) {
        if ((col - x) * (col - x) + (row - y) * (row - y) <= DISTANCE * DISTANCE)
        {
            int pixVal_r = 0;
            int pixVal_g = 0;
            int pixVal_b = 0;
            int pixels = 0;
            // Get the average of the surrounding pixels
            for (int fRow = -BLUR_SIZE; fRow <= BLUR_SIZE; fRow++) {
                for (int fCol = -BLUR_SIZE; fCol <= BLUR_SIZE; fCol++) {
                    int tileRow = threadIdx.y + fRow;
                    int tileCol = threadIdx.x + fCol;
                    if (tileRow >= 0 && tileRow < TILE_DIM && tileCol >= 0 && tileCol < TILE_DIM) {
                        pixVal_r += dr_in_shared [tileRow][tileCol];
                        pixVal_g += dg_in_shared [tileRow][tileCol];
                        pixVal_b += db_in_shared [tileRow][tileCol];
                        ++pixels;
                    }
                    else {
                        int i = row + fRow;
                        int j = col + fCol;
                        if (i >= 0 && i < height && j >= 0 && j < width) {
                            pixVal_r += dr_in[idx * numPixels + i * width + j];
                            pixVal_g += dg_in[idx * numPixels + i * width + j];
                            pixVal_b += db_in[idx * numPixels + i * width + j];
                            ++pixels;
                        }
                    }
                }
            }
            //printf("%lf %lf %lf\n", (double)pixVal_r/pixels, (double)pixVal_g/pixels, (double)pixVal_b/pixels);
            dr_out [idx * numPixels + row * width + col] = (uchar) (pixVal_r / pixels);
            dg_out [idx * numPixels + row * width + col] = (uchar) (pixVal_g / pixels);
            db_out [idx * numPixels + row * width + col] = (uchar) (pixVal_b / pixels);
        }
        else
        {
            dr_out [idx * numPixels + row * width + col] = dr_in [idx * numPixels + row * width + col];
            dg_out [idx * numPixels + row * width + col] = dg_in [idx * numPixels + row * width + col];
            db_out [idx * numPixels + row * width + col] = db_in [idx * numPixels + row * width + col];
        }
    }
    //printf("%d\n", idx);
}

__host__ void copyFromHostToDevice (uchar *hr_in, uchar *hg_in, uchar *hb_in, uchar *dr_in, uchar *dg_in, uchar *db_in, int cnt, int width, int height) {
    int numPixels = cnt * width * height;
    size_t size = numPixels * sizeof (uchar);
    cudaMemcpy(dr_in, hr_in, size, cudaMemcpyHostToDevice);
    check("Error in cudaMemcpy host to device");
    cudaMemcpy(dg_in, hg_in, size, cudaMemcpyHostToDevice);
    check("Error in cudaMemcpy host to device");
    cudaMemcpy(db_in, hb_in, size, cudaMemcpyHostToDevice);
    check("Error in cudaMemcpy host to device");
}

__host__ void copyFromDeviceToHost (uchar *dr_out, uchar *dg_out, uchar *db_out, uchar *hr_out, uchar *hg_out, uchar *hb_out, int cnt, int width, int height) {
    int numPixels = cnt * width * height;
    size_t size = numPixels * sizeof (uchar);
    cudaMemcpy(hr_out, dr_out, size, cudaMemcpyDeviceToHost);
    check("Error in cudaMemcpy device to host dr_out");
    cudaMemcpy(hg_out, dg_out, size, cudaMemcpyDeviceToHost);
    check("Error in cudaMemcpy device to host dg_out");
    cudaMemcpy(hb_out, db_out, size, cudaMemcpyDeviceToHost);
    check("Error in cudaMemcpy device to host db_out");
}

__host__ void freeDeviceMemory (uchar *dr_in, uchar *dg_in, uchar *db_in, uchar *dr_out, uchar *dg_out, uchar *db_out) {
    cudaFree(dr_in);
    check("Error in cudaFree dr_in");
    cudaFree(dg_in);
    check("Error in cudaFree dg_in");
    cudaFree(db_in);
    check("Error in cudaFree db_in");
    cudaFree(dr_out);
    check("Error in cudaFree dr_out");
    cudaFree(dg_out);
    check("Error in cudaFree dg_out");
    cudaFree(db_out);
    check("Error in cudaFree db_out");
}

__host__ void cleanUp () {
    cudaDeviceReset();
    check("Error in cudaDeviceReset");
}

__global__ void blur (uchar *dr_in, uchar *dg_in, uchar *db_in, uchar *dr_out, uchar *dg_out, uchar *db_out, int width, int height, int x, int y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dim3 blockSize (TILE_DIM, TILE_DIM);
    dim3 gridSize ((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    convert <<<gridSize, blockSize>>> (dr_in, dg_in, db_in, dr_out, dg_out, db_out, idx, width, height, x, y);
}

__host__ void execute (uchar *dr_in, uchar *dg_in, uchar *db_in, uchar *dr_out, uchar *dg_out, uchar *db_out, int cnt, int width, int height, int x, int y) {
    blur <<<1, cnt>>> (dr_in, dg_in, db_in, dr_out, dg_out, db_out, width, height, x, y);
    check("Error in kernel call");
    cudaDeviceSynchronize ();
}

__host__ void readPartImageFromFile(Mat* image, uchar* h_r, uchar* h_g, uchar* h_b, int width, int height, int idx_x, int idx_y, int interval_X, int interval_Y) {
    for (int i = idx_y * interval_Y; i < min((idx_y + 1) * interval_Y, height); i++) {
        for (int j = idx_x * interval_X; j < min((idx_x + 1) * interval_X, width); j++) {
            Vec3b pixel = image->at<Vec3b> (i, j);
            uchar blue = pixel.val[0];
            uchar green = pixel.val[1];
            uchar red = pixel.val[2];
            h_r [i * width + j] = red;
            h_g [i * width + j] = green;
            h_b [i * width + j] = blue; 
        }
    }
}

__host__ void readImageFromFile(Mat* image, uchar* hr_total, uchar* hg_total, uchar* hb_total, int cnt, int width, int height) {
    int numPixels = width * height;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            Vec3b pixel = image->at<Vec3b> (i, j);
            uchar blue = pixel.val[0];
            uchar green = pixel.val[1];
            uchar red = pixel.val[2];
            
            hr_total [cnt * numPixels + i * width + j] = red;
            hg_total [cnt * numPixels + i * width + j] = green;
            hb_total [cnt * numPixels + i * width + j] = blue;
        }
    }
}

void onMouse(int event, int x, int y, int, void* userdata) {
    Mat* image = reinterpret_cast<cv::Mat*>(userdata);
    if (event == EVENT_LBUTTONDOWN) {
        blur_x = x;
        blur_y = y;
    }
    else if (event == EVENT_RBUTTONDOWN) {
        blur_x = -1;
        blur_y = -1;
    }
}

int main(int argc, char** argv) 
{
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <video_file_path>" << endl;
        return -1;
    }

    string videoFilePath = argv[1];

    try
    {
        VideoCapture cap(videoFilePath);
        if (!cap.isOpened())
        {
            cout << "Error: Video file not opened\n";
            return -1;
        }

        int width = cap.get(CAP_PROP_FRAME_WIDTH);
        int height = cap.get(CAP_PROP_FRAME_HEIGHT);
        int frames = cap.get(CAP_PROP_FPS);
        int numPixels = width * height;

        cout << "Width: " << width << " Height: " << height << " Frames: " << frames << endl;

        uchar *dr_in, *dg_in, *db_in, *dr_out, *dg_out, *db_out;
        cudaMalloc(&dr_in, numPixels * NUM_FRAMES * sizeof(uchar));
        check("Error in cudaMalloc dr_in");
        cudaMalloc(&dg_in, numPixels * NUM_FRAMES * sizeof(uchar));
        check("Error in cudaMalloc dg_in");
        cudaMalloc(&db_in, numPixels * NUM_FRAMES * sizeof(uchar));
        check("Error in cudaMalloc db_in");
        cudaMalloc(&dr_out, numPixels * NUM_FRAMES * sizeof(uchar));
        check("Error in cudaMalloc dr_out");
        cudaMalloc(&dg_out, numPixels * NUM_FRAMES * sizeof(uchar));
        check("Error in cudaMalloc dg_out");
        cudaMalloc(&db_out, numPixels * NUM_FRAMES * sizeof(uchar));
        check("Error in cudaMalloc db_out");

        uchar *hr_in = (uchar *) malloc (numPixels * NUM_FRAMES * sizeof (uchar));
        uchar *hg_in = (uchar *) malloc (numPixels * NUM_FRAMES * sizeof (uchar));
        uchar *hb_in = (uchar *) malloc (numPixels * NUM_FRAMES * sizeof (uchar));

        uchar *hr_out = (uchar *) malloc (numPixels * NUM_FRAMES * sizeof (uchar));
        uchar *hg_out = (uchar *) malloc (numPixels * NUM_FRAMES * sizeof (uchar));
        uchar *hb_out = (uchar *) malloc (numPixels * NUM_FRAMES * sizeof (uchar));

        Mat frame;

        auto start = chrono::high_resolution_clock::now();

        while(true)
        {
            int cnt = 0;
            bool flag = true;
            for(int i = 0; i < NUM_FRAMES; i++)
            {
                if (!cap.read(frame)) {
                    flag = false;
                    cout << "Error: Video file not opened\n";
                    auto end = chrono::high_resolution_clock::now();
                    chrono::duration<float, std::milli> duration_ms = end - start;
                    cout << "Time: " << duration_ms.count() << " ms" << endl;
                }
                readImageFromFile (&frame, hr_in, hg_in, hb_in, cnt, width, height);
                ++cnt;
            }

            setMouseCallback("Blurred Image", onMouse, &frame);
            cout << blur_x << " " << blur_y << endl;
            if (blur_x == -1 && blur_y == -1) {
                Mat outputImage = Mat::zeros (height, width, CV_8UC3);
                for(int idx = 0; idx < cnt; idx++)
                {
                    
                    #pragma omp parallel for collapse(2)
                    for (int i = 0; i < height; i++) {
                        for (int j = 0; j < width; j++) {
                            Vec3b pixel;
                            pixel.val[2] = hr_in [idx * numPixels + i * width + j];
                            pixel.val[1] = hg_in [idx * numPixels + i * width + j];
                            pixel.val[0] = hb_in [idx * numPixels + i * width + j];
                            outputImage.at<Vec3b> (i, j) = pixel;
                        }
                    }
                    imshow("Blurred Image", outputImage);
                    if( waitKey(1000 / frames) == 27 )
                    {
                        flag = false;
                        break;
                    }
                }
            }

            else {
                copyFromHostToDevice (hr_in, hg_in, hb_in, dr_in, dg_in, db_in, cnt, width, height);
                check("Error in copyFromHostToDevice");
                execute(dr_in, dg_in, db_in, dr_out, dg_out, db_out, cnt, width, height, blur_x, blur_y);
                check("Error in execute");
                copyFromDeviceToHost (dr_out, dg_out, db_out, hr_out, hg_out, hb_out, cnt, width, height);
                check("Error in copyFromDeviceToHost");
                
                Mat outputImage = Mat::zeros (height, width, CV_8UC3);
                for(int idx = 0; idx < cnt; idx++)
                {
                    
                    #pragma omp parallel for collapse(2)
                    for (int i = 0; i < height; i++) {
                        for (int j = 0; j < width; j++) {
                            Vec3b pixel;
                            pixel.val[2] = hr_out [idx * numPixels + i * width + j];
                            pixel.val[1] = hg_out [idx * numPixels + i * width + j];
                            pixel.val[0] = hb_out [idx * numPixels + i * width + j];
                            outputImage.at<Vec3b> (i, j) = pixel;
                        }
                    }
                    imshow("Blurred Image", outputImage);
                    if( waitKey(1000 / frames) == 27 )
                    {
                        flag = false;
                        break;
                    }
                }
            }
            if(flag == false) break;
        }

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<float, std::milli> duration_ms = end - start;
        cout << "Time: " << duration_ms.count() << " ms" << endl;
    }
    catch (const exception &e)
    {
        cout << "Error: " << e.what () << endl;
        return 1;
    }
    return 0;
}