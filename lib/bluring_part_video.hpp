#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <omp.h>
#include <bits/stdc++.h>
#include <chrono>
#include <thread>
#include <barrier>

#define BLUR_SIZE 70
#define TILE_DIM 32
#define DISTANCE 300
#define NUM_FRAMES 16

using namespace cv;
using namespace std;
