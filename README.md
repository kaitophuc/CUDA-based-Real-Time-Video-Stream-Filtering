# CUDA-based Real-Time Video Stream Filtering

## Overview

This project leverages **CUDA** technology to apply real-time blurring to specified regions of video frames with automatic **face detection**. The primary objective is to demonstrate how **GPU acceleration** can be used for video processing tasks like selective blurring, useful in various scenarios such as privacy masking, content filtering, or artistic effects.

Developed as part of a final assignment for a Coursera course, this project showcases the efficiency of **GPU computing** in handling large-scale data operations, particularly for time-sensitive applications like real-time video editing. The project features multiple CUDA kernel implementations for performance comparison and optimization analysis.

## Key Features

- **Real-time face detection** using OpenCV's DNN module with SSD MobileNet
- **Multiple CUDA kernel implementations** for performance comparison
- **Comprehensive benchmarking framework** for kernel testing
- **Interactive and automated testing modes**
- **Webcam support** for live video processing
- **GPU-accelerated blurring** with stream-based parallel processing
- **Memory-optimized CUDA operations** with proper cleanup

## Project Structure

```
├── data/               # Video files and test datasets
│   └── input.mp4      # Sample video file
├── src/               # Source code
│   └── bluring_part_video.cu  # Main CUDA implementation
├── lib/               # Header files
│   └── bluring_part_video.hpp # Function declarations
├── bin/               # Compiled binaries
├── models/            # AI models for face detection
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   └── shape_predictor_68_face_landmarks.dat
├── Makefile          # Build configuration
└── run.sh            # Comprehensive execution script with testing capabilities
```

## Available CUDA Kernels

The project implements multiple kernel variants for performance comparison:

### 1. Naive CUDA Kernel
- Basic CUDA implementation with simple streams
- Standard memory management
- RGB channel processing with basic parallelization

### 2. Optimized CUDA Kernel
- Improved memory access patterns
- Enhanced stream management for better concurrency
- Optimized data transfer operations
- Better resource utilization

## Kernel Performance Testing Framework

### Usage Modes

#### Quick Start
```bash
# Build the project
make build

# Interactive mode selection (recommended for beginners)
./run.sh

# Direct execution with webcam benchmark
./run.sh 0 webcam_benchmark

# Build and run webcam benchmark
./run.sh --build 0 webcam_benchmark

# Interactive mode with video file
./run.sh data/input.mp4 interactive

# Test mode with 100 frames
./run.sh data/input.mp4 test

# Show all available options
./run.sh --help
```

#### Available Modes

**Test Mode (`test`)**
- Tests all available kernels with 100 frames
- Shows performance comparison table
- Identifies the best performing kernel

**Interactive Mode (`interactive`)**
- Choose which kernel to use for real-time processing
- Real-time face detection and blurring
- Left click to enable blur, right click to disable
- ESC to exit

**Webcam Benchmark Mode (`webcam_benchmark`)**
- Tests all kernels with live webcam feed
- 10-second benchmark per kernel
- Real-time FPS measurement and comparison
- Automatic best kernel identification

### Performance Metrics

The framework measures:
- **Average FPS**: Frames processed per second
- **Total Time**: Total processing time in seconds
- **Frame Count**: Number of frames processed
- **Memory Usage**: CUDA memory allocation efficiency

### Example Performance Results

```
=== WEBCAM BENCHMARK SUMMARY ===
         Kernel Name        Avg FPS   Total Frames  Test Duration
-----------------------------------------------------------------
          Naive CUDA          59.50            596           10.0s
      Optimized CUDA          60.49            605           10.0s

Best performing kernel: Optimized CUDA (60.49 FPS)
```

## Dependencies

- **CUDA Toolkit** (11.0 or higher)
- **OpenCV 4.x** with DNN module support
- **OpenMP** for CPU parallelization
- **CMake** or **Make** for building
- **GCC/G++** compiler with C++20 support

## Prerequisites

Before building and running the project, ensure the following are installed:

1. **NVIDIA GPU** with CUDA capability 3.5 or higher
2. **CUDA Toolkit**: Download from [NVIDIA's CUDA Toolkit page](https://developer.nvidia.com/cuda-toolkit)
3. **OpenCV**: Install with CUDA support enabled
4. **OpenMP**: Usually included with GCC
5. **Webcam** (optional, for live testing)

## Build Instructions

### Linux/Ubuntu

```bash
# Clone the repository
git clone <repository-url>
cd project_CUDA_based_Real_Time_Video_Stream_Filtering

# Make the run script executable
chmod +x run.sh

# Build and run with interactive mode selection
./run.sh --build --interactive

# Or build manually and run specific mode
make build
./run.sh 0 webcam_benchmark
```

### Windows (using WSL)

1. Install Windows Subsystem for Linux (WSL2)
2. Install CUDA toolkit in WSL
3. Follow the Linux instructions above

## Advanced Usage

### Adding New Kernels

To implement and test a new kernel:

1. **Implement the kernel function** with this signature:
```cpp
void YourKernel(cv::Mat& frame, int width, int height, int frames, int num_pixels,
               uchar* hr_in, uchar* hg_in, uchar* hb_in, 
               uchar* hr_out, uchar* hg_out, uchar* hb_out,
               uchar* dr_in, uchar* dg_in, uchar* db_in, 
               uchar* dr_out, uchar* dg_out, uchar* db_out);
```

2. **Add it to the kernels vector** in main():
```cpp
std::vector<KernelPerformance> kernels = {
    KernelPerformance("Naive CUDA", Blur_Naive),
    KernelPerformance("Optimized CUDA", Blur_Optimized),
    KernelPerformance("Your Kernel", YourKernel)  // Add here
};
```

### Kernel Development Tips

1. **Use CUDA streams** for parallel RGB channel processing
2. **Optimize memory access patterns** for better coalescing
3. **Consider shared memory** for frequently accessed data
4. **Profile with nvprof/nsight** for detailed performance analysis
5. **Test with different resolutions** to understand scalability
6. **Implement proper error checking** for robust operation

### Performance Optimization

- **Memory Management**: Use unified memory for simplified allocation
- **Stream Processing**: Parallel RGB channel operations
- **Kernel Launch Parameters**: Optimize block and grid sizes
- **Memory Coalescing**: Ensure optimal memory access patterns
- **Occupancy**: Balance threads per block with register usage

## Technical Implementation

### Face Detection Pipeline
1. **Frame Capture**: OpenCV VideoCapture for input
2. **DNN Processing**: SSD MobileNet for face detection
3. **Coordinate Extraction**: Bounding box calculation
4. **CUDA Processing**: GPU-accelerated blur application
5. **Frame Display**: Real-time visualization

### Memory Architecture
- **Unified Memory**: Simplified CPU-GPU data sharing
- **Host Memory**: Pinned memory for faster transfers
- **Device Memory**: GPU global memory for processing
- **Stream Management**: Asynchronous operations

## Troubleshooting

### Common Issues

**Build Errors:**
- Ensure CUDA toolkit is properly installed
- Check OpenCV installation and CUDA support
- Verify compiler compatibility (GCC 9+ recommended)

**Runtime Issues:**
- Check webcam permissions and availability
- Verify model files are present in `models/` directory
- Ensure sufficient GPU memory (2GB+ recommended)

**Performance Issues:**
- Monitor GPU memory usage with `nvidia-smi`
- Check for thermal throttling
- Verify optimal block/grid sizes for your GPU

### Memory Management
The project implements robust memory cleanup:
- Automatic resource deallocation
- Stream synchronization before cleanup
- Proper error handling for CUDA operations
- Prevention of memory leaks

## Results and Performance

**Real-time Performance**: Achieves 60+ FPS on modern GPUs (GTX 1060 or better)

**Optimization Impact**: The optimized CUDA kernel shows ~1-2% improvement over naive implementation

**Memory Efficiency**: Unified memory reduces code complexity while maintaining performance

**Scalability**: Performance scales well with input resolution and GPU capability

This project demonstrates the effectiveness of GPU acceleration for real-time video processing tasks, with a focus on practical implementation and performance optimization techniques.