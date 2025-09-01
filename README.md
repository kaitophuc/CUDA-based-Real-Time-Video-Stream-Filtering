# CUDA-based Real-Time Video Stream Filtering

## Overview

This project implements CUDA-accelerated face detection and blurring for real-time video processing. It compares three different CUDA kernel implementations for selective blurring of detected faces in video streams or webcam feeds.

The project features automatic face detection using OpenCV's DNN module with SSD MobileNet and applies GPU-accelerated blur effects to detected facial regions. Multiple CUDA kernel implementations are available for performance comparison and educational purposes.

## Project Structure

```
├── data/               # Video files for testing
│   └── input.mp4      # Sample video files
├── src/               # Source code
│   └── bluring_part_video.cu  # Main CUDA implementation
├── lib/               # Header files and CUDA kernels
│   └── bluring_part_video.hpp # CUDA kernel implementations
├── bin/               # Compiled binaries
├── models/            # AI models for face detection
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   └── shape_predictor_68_face_landmarks.dat
├── Makefile          # Build configuration
└── run.sh            # Execution script
```

## Available CUDA Kernels

The project implements three different CUDA kernel variants for performance comparison:

### 1. **Naive CUDA Kernel**
- Single kernel launch processing all RGB channels together
- Uses shared memory tiling (32x32 tiles)
- Sequential memory transfers
- Blur radius: 128 pixels

### 2. **Multi-Stream CUDA Kernel**
- Three separate kernel launches (one per RGB channel)
- Uses three CUDA streams for parallel processing
- Supports up to 3 faces simultaneously
- Per-channel blur processing with shared memory

### 3. **CUB Optimized Kernel**
- Uses NVIDIA CUB library for optimized block operations
- Separable box blur (horizontal + vertical passes)
- Advanced grid configurations for better occupancy
- Reduced memory allocations with temporary buffers

## Features

- **Real-time face detection** using OpenCV's DNN module
- **GPU-accelerated blurring** with multiple kernel implementations
- **Multi-face support** (up to 3 faces simultaneously)
- **Performance benchmarking** framework with FPS measurement
- **Webcam and video file support**
- **Interactive testing modes**
- **Exponential moving average** for smooth FPS display

## Usage Modes

### Quick Start
```bash
# Build the project
make build

# Interactive mode (choose kernel)
./run.sh 0 interactive

# Test all kernels with video file
./run.sh data/input.mp4 test

# Benchmark with webcam
./run.sh 0 webcam_benchmark
```

### Available Modes

**Test Mode (`test`)**
- Tests all kernels with entire video file
- Processes all frames for comprehensive comparison
- Outputs performance summary table

**Interactive Mode (`interactive`)**
- Choose specific kernel for real-time processing
- Left click to enable blur, right click to disable
- ESC to exit

**Webcam Benchmark Mode (`webcam_benchmark`)**
- 10-second benchmark per kernel
- Real-time progress display
- Automatic best kernel identification

**Benchmark Mode (`benchmark`)**
- Tests all kernels then runs best one interactively

### Command Line Usage

```bash
# Usage pattern
./run.sh <video_source> <mode>

# Examples
./run.sh 0 interactive                    # Webcam interactive
./run.sh data/video.mp4 test             # Video file testing
./run.sh 0 webcam_benchmark              # Webcam benchmarking
```

## Performance Metrics

The framework measures:
- **Average FPS**: Overall frames per second
- **Total Time**: Complete processing duration
- **Frame Count**: Number of frames processed
- **Smoothed FPS**: Exponential moving average for display

## Dependencies

- **CUDA Toolkit** (11.0+)
- **OpenCV 4.x** with DNN module
- **OpenMP** for CPU parallelization
- **CUB Library** (included with CUDA 11.0+)
- **cuDNN** (optional, for advanced optimizations)

## Build Instructions

### Linux/Ubuntu

```bash
# Install dependencies
sudo apt update
sudo apt install libopencv-dev libopencv-contrib-dev

# Clone and build
git clone <repository-url>
cd project_CUDA_based_Real_Time_Video_Stream_Filtering
make build

# Run with webcam
./run.sh 0 interactive
```

## Technical Implementation

### Face Detection Pipeline
1. Frame capture via OpenCV VideoCapture
2. DNN inference using SSD MobileNet (300x300 input)
3. Confidence filtering (>0.5) and coordinate extraction
4. Top 3 faces selected by confidence score
5. CUDA kernel application for selective blurring

### Memory Architecture
- **Unified Memory**: Face coordinates and detection data
- **Pinned Host Memory**: Faster CPU-GPU transfers
- **Device Memory**: RGB channel processing buffers
- **Shared Memory**: Block-level cache (32x32 tiles)

### CUDA Implementation Details
- **Block Size**: 32x32 threads per block
- **Memory Transfers**: Asynchronous for RGB channels  
- **Blur Algorithm**: Box filter with configurable radius
- **Stream Management**: Single stream (Naive) vs. Triple stream (Multi-Stream)

## Controls

### Interactive Mode
- **Left Click**: Enable blur at clicked position
- **Right Click**: Disable all blur effects
- **ESC**: Exit application

### Face Detection Mode
- Automatic detection of up to 3 faces
- Blur applied to circular regions around face centers
- Real-time bounding box visualization

## Performance Results

Actual benchmark results from running the project on a video file (2999 frames, 1920x1080 @ 29 FPS):

```
=== KERNEL PERFORMANCE COMPARISON ===

=== Testing Kernel: Naive CUDA ===
Processing entire video...
Frames processed: 2999
Total time: 55.516 seconds
Average FPS: 54.0205
Final smoothed FPS: 50.0818

=== Testing Kernel: Multi-Stream CUDA ===
Processing entire video...
Frames processed: 2999
Total time: 78.334 seconds
Average FPS: 38.2848
Final smoothed FPS: 33.3488

=== Testing Kernel: CUB Optimized ===
Processing entire video...
Frames processed: 2999
Total time: 36.518 seconds
Average FPS: 82.1239
Final smoothed FPS: 102.523

=== PERFORMANCE SUMMARY ===
         Kernel Name        Avg FPS     Total Time
--------------------------------------------------
          Naive CUDA          54.02          55.52s
   Multi-Stream CUDA          38.28          78.33s
       CUB Optimized          82.12          36.52s

Best performing kernel: CUB Optimized (82.12 FPS)
```

**Key Findings:**
- **CUB Optimized** performs best with 82.12 FPS (52% faster than Naive)
- **Multi-Stream CUDA** is actually slower than Naive due to stream overhead
- **Naive CUDA** provides good baseline performance at 54.02 FPS
- Processing a 1080p video with face detection and blurring in real-time