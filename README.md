# CUDA-based Real-Time Video Stream Filtering

## Overview

This project implements CUDA-accelerated face detection and blurring for real-time video processing. It features a **modular architecture** that separates each CUDA kernel implementation into dedicated files, making the codebase more maintainable and easier to understand. The project compares four different CUDA kernel implementations for selective blurring of detected faces in video streams or webcam feeds.

The project features automatic face detection using OpenCV's DNN module with SSD MobileNet and applies GPU-accelerated blur effects to detected facial regions. Multiple CUDA kernel implementations are available for performance comparison and educational purposes.

## Modular Project Structure

The codebase has been reorganized into a modular architecture where each blur kernel is implemented in separate files:

```
├── data/               # Video files for testing
│   ├── input.mp4      # Sample video files
│   └── merged_clips.mp4
├── src/               # Source code (modular organization)
│   ├── bluring_part_video.cu    # Main application orchestrator
│   ├── blur_common.cu           # Common utilities and functions
│   ├── blur_naive.cu            # Naive CUDA kernel implementation
│   ├── blur_multistream.cu      # Multi-stream CUDA implementation
│   ├── blur_cub.cu              # CUB optimized implementation
│   ├── blur_prefix_sum.cu       # Brent-Kung prefix sum implementation
│   └── blur_thrust.cu           # Thrust library implementation
├── lib/               # Header files (modular organization)
│   ├── blur_common.hpp          # Common definitions and utilities
│   ├── blur_naive.hpp           # Naive kernel declarations
│   ├── blur_multistream.hpp     # Multi-stream kernel declarations
│   ├── blur_cub.hpp             # CUB kernel declarations
│   ├── blur_prefix_sum.hpp      # Brent-Kung prefix sum declarations
│   ├── blur_thrust.hpp          # Thrust library declarations
│   └── bluring_part_video.hpp   # Legacy header (compatibility)
├── bin/               # Compiled binaries
│   └── bluring_part_video.exe
├── models/            # AI models for face detection
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   └── shape_predictor_68_face_landmarks.dat
├── tools/             # Utility scripts for video processing
│   ├── extract_clips.py         # Extract random clips from videos
│   └── merge_clips.py           # Merge multiple clips into one video
├── Makefile          # Updated build configuration for modular build
└── run.sh            # Comprehensive execution script
```

## Modular Architecture Benefits

### 1. **Separation of Concerns**
- Each blur kernel is implemented in its own dedicated file
- Common functionality is centralized in `blur_common.cu/hpp`
- Main application focuses on orchestration and user interface

### 2. **Maintainability**
- Easy to modify individual kernels without affecting others
- Clear separation between kernel implementations and common utilities
- Simplified debugging and testing of specific components

### 3. **Extensibility**
- New blur kernels can be added by creating new files and updating the kernel vector
- Common functions are reusable across all implementations
- Consistent interface for all kernel implementations

### 4. **Educational Value**
- Each kernel implementation can be studied independently
- Clear comparison between different CUDA programming techniques
- Modular structure makes it easier to understand complex CUDA concepts

## Available CUDA Kernels

The project implements four different CUDA kernel variants, each in its own file:

### 1. **Naive CUDA Kernel** (`src/blur_naive.cu`)
- **Architecture**: Single kernel launch processing all RGB channels together
- **Memory Management**: Sequential memory transfers with single stream
- **Optimization**: Uses shared memory tiling (32x32 tiles)
- **Blur Method**: Box filter with 128-pixel radius
- **Use Case**: Baseline implementation for comparison

### 2. **Multi-Stream CUDA Kernel** (`src/blur_multistream.cu`)
- **Architecture**: Three separate kernel launches (one per RGB channel)
- **Memory Management**: Uses three CUDA streams for parallel processing
- **Optimization**: Asynchronous memory transfers with stream parallelism
- **Multi-Face Support**: Processes up to 3 faces simultaneously
- **Use Case**: Demonstrates stream parallelism benefits

### 3. **CUB Optimized Kernel** (`src/blur_cub.cu`)
- **Architecture**: Uses NVIDIA CUB library for optimized block operations
- **Memory Management**: Separable box blur (horizontal + vertical passes)
- **Optimization**: Advanced grid configurations and block-level reductions
- **Performance**: Reduced memory allocations with temporary buffers
- **Use Case**: High-performance production implementation

### 4. **Brent-Kung Prefix Sum Kernel** (`src/blur_prefix_sum.cu`)
- **Architecture**: Custom implementation using Brent-Kung algorithm for prefix sums
- **Memory Management**: Separable box blur with hand-crafted prefix sum operations
- **Optimization**: Work-efficient O(n) prefix sum algorithm
- **Educational Value**: Demonstrates custom parallel algorithm implementation
- **Use Case**: Research and educational comparison with CUB library

### 5. **Thrust Library Kernel** (`src/blur_thrust.cu`)
- **Architecture**: High-level parallel algorithms using CUDA Thrust library
- **Memory Management**: Host-side Thrust operations with device kernel coordination
- **Optimization**: STL-like interface with automatic algorithm selection
- **Algorithm**: Two-pass separable blur using `thrust::inclusive_scan` for prefix sums
- **Benefits**: More readable code, portable across different GPU architectures
- **Use Case**: Rapid prototyping and high-level CUDA programming

## Common Utilities (`src/blur_common.cu`)

The common utilities module provides shared functionality:
- **Memory Management**: Host and device memory allocation functions
- **Face Detection**: OpenCV DNN integration with SSD MobileNet
- **Performance Measurement**: FPS calculation with exponential moving average
- **Video Processing**: Frame reading and processing utilities
- **Testing Framework**: Kernel benchmarking and interactive modes

## Features

- **Real-time face detection** using OpenCV's DNN module
- **GPU-accelerated blurring** with multiple kernel implementations
- **Multi-face support** (up to 3 faces simultaneously)
- **Performance benchmarking** framework with FPS measurement
- **Webcam and video file support**
- **Interactive testing modes**
- **Exponential moving average** for smooth FPS display
- **Modular architecture** for easy maintenance and extension

## Usage - Single Command Execution

### Quick Start
```bash
# Build the project
make build

# Test all kernels with video file
./run.sh data/input.mp4 test

# Run specific kernel interactively with webcam
./run.sh 0 interactive naive
./run.sh 0 interactive multistream
./run.sh 0 interactive cub
./run.sh 0 interactive brentkunng
./run.sh 0 interactive thrust

# Run specific kernel with video file
./run.sh data/input.mp4 interactive cub
./run.sh data/input.mp4 interactive thrust
```

### Command Syntax
```bash
./run.sh [video_source] [mode] [kernel]
```

**Parameters:**
- `video_source`: Video file path or `0` for webcam
- `mode`: `test` or `interactive`
- `kernel`: `naive`, `multistream`, `cub`, `brentkunng`, or `thrust` (required for interactive mode)

### Available Modes

**Test Mode (`test`)**
- Tests all kernels sequentially
- Processes entire video for comprehensive comparison
- Outputs performance summary table
- Example: `./run.sh data/input.mp4 test`

**Interactive Mode (`interactive`)**
- Runs specified kernel in real-time
- Left click to enable blur, right click to disable
- ESC to exit
- Requires kernel selection
- Example: `./run.sh 0 interactive cub`

### Available Kernels

- **`naive`**: Naive CUDA implementation (baseline)
- **`multistream`**: Multi-stream parallel processing
- **`cub`**: CUB library optimized version
- **`brentkunng`**: Brent-Kung prefix sum algorithm
- **`thrust`**: Thrust Library implementation

### Examples

```bash
# Test all kernels with webcam (10 seconds each)
./run.sh 0 test

# Run CUB kernel interactively with webcam
./run.sh 0 interactive cub

# Test all kernels with video file
./run.sh data/merged_clips.mp4 test

# Run Brent-Kung kernel with video file
./run.sh data/input.mp4 interactive brentkunng

# Run Thrust kernel with webcam
./run.sh 0 interactive thrust
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
- **Thrust Library** (included with CUDA Toolkit)
- **cuDNN** (optional, for advanced optimizations)

## Build Instructions

### Linux/Ubuntu

```bash
# Install dependencies
sudo apt update
sudo apt install libopencv-dev libopencv-contrib-dev python3 python3-opencv

# Clone and build
git clone <repository-url>
cd project_CUDA_based_Real_Time_Video_Stream_Filtering
make build

# Run with webcam
./run.sh 0 interactive
```

## Technical Implementation

### Modular Architecture Details

#### **Main Application** (`src/bluring_part_video.cu`)
- Orchestrates all kernel implementations
- Handles command-line argument parsing
- Manages memory allocation and cleanup
- Provides user interface for kernel selection

#### **Common Utilities** (`src/blur_common.cu`)
- Face detection pipeline with OpenCV DNN
- Memory management for host and device
- Performance measurement and FPS calculation
- Video capture and frame processing utilities

#### **Kernel Implementations**
- **Naive**: `src/blur_naive.cu` - Simple single-kernel approach
- **Multi-Stream**: `src/blur_multistream.cu` - Stream parallelism
- **CUB Optimized**: `src/blur_cub.cu` - Advanced CUB operations
- **Brent-Kung**: `src/blur_prefix_sum.cu` - Custom prefix sum algorithm
- **Thrust**: `src/blur_thrust.cu` - High-level parallel algorithms with STL-like interface

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
- **Kernel Selection**: Specified via command line (e.g., `./run.sh 0 interactive cub`)

### Automatic Face Detection
- Detects up to 3 faces automatically
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

=== Testing Kernel: Brent-Kung Prefix Sum ===
Processing entire video...
Frames processed: 2999
Total time: 34.905 seconds
Average FPS: 85.9189
Final smoothed FPS (realtime): 115.367

=== PERFORMANCE SUMMARY ===
         Kernel Name        Avg FPS   Realtime FPS     Total Time
-----------------------------------------------------------------
          Naive CUDA          51.28          51.36          58.49s
   Multi-Stream CUDA          37.38          32.23          80.23s
       CUB Optimized          86.25         111.69          34.77s
Brent-Kung Prefix Sum          85.92         115.37          34.91s

Best performing kernel: Brent-Kung Prefix Sum (Realtime: 115.37 FPS, Average: 85.92 FPS)
```

**Key Findings:**
- **CUB Optimized** performs best with 82.12 FPS (52% faster than Naive)
- **Brent-Kung Prefix Sum** competitive with CUB while being custom implementation
- **Multi-Stream CUDA** is slower than Naive due to stream overhead with this workload
- **Naive CUDA** provides good baseline performance at 54.02 FPS
- Processing a 1080p video with face detection and blurring in real-time
- Custom algorithms can approach library performance with proper optimization

## Development Notes

### Adding New Kernels

To add a new blur kernel implementation:

1. **Create header file**: `lib/blur_newkernel.hpp`
```cpp
#ifndef BLUR_NEWKERNEL_HPP
#define BLUR_NEWKERNEL_HPP
#include "blur_common.hpp"

void Blur_NewKernel(cv::Mat& frame, int width, int height, int frames, int num_pixels, 
                   uchar* hr_in, uchar* hg_in, uchar* hb_in, 
                   uchar* hr_out, uchar* hg_out, uchar* hb_out,
                   uchar* dr_in, uchar* dg_in, uchar* db_in, 
                   uchar* dr_out, uchar* dg_out, uchar* db_out);

#endif
```

2. **Create implementation file**: `src/blur_newkernel.cu`
3. **Update main application**: Add include and register kernel
4. **Update Makefile**: Add new source file to build process

### Debugging Tips

- Use `CheckCudaError()` after CUDA operations for error checking
- Enable debug prints in common utilities for tracing execution
- Use NVIDIA Nsight Systems for performance profiling
- Each kernel can be tested independently in interactive mode