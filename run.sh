#!/bin/bash

# CUDA-based Real-Time Video Stream Filtering
# Single-command execution script

echo "=== CUDA-based Real-Time Video Stream Filtering ==="
echo ""

# Directory containing input files
DATA_DIR="./data"

# Function to display usage information
show_usage() {
    echo "Usage: $0 [OPTIONS] [video_source] [kernel]"
    echo ""
    echo "OPTIONS:"
    echo "  --help     Show this help message"
    echo "  --build    Build the project before running"
    echo ""
    echo "Video sources:"
    echo "  0              Use webcam"
    echo "  video.mp4      Use video file from data/ directory"
    echo "  full/path.mp4  Use video file with full path"
    echo ""
    echo "Kernels:"
    echo "  naive          Naive CUDA implementation"
    echo "  multistream    Multi-Stream CUDA implementation"
    echo "  cub            CUB Optimized implementation"
    echo "  brentkunng     Brent-Kung Prefix Sum implementation"
    echo "  thrust         Thrust Library implementation"
    echo "  test           Test and compare all kernels"
    echo "  singletest     Test a single kernel with performance metrics"
    echo ""
    echo "Examples:"
    echo "  $0 0 cub                     # Webcam + CUB kernel (interactive)"
    echo "  $0 video.mp4 brentkunng      # Video + Brent-Kung kernel (interactive)"
    echo "  $0 --build 0 test            # Build + Webcam + test all kernels"
    echo "  $0 0 thrust                  # Webcam + Thrust kernel (interactive)"
    echo "  $0 0 singletest thrust       # Test only Thrust kernel with performance metrics"
    echo "  $0 data/input.mp4 singletest naive # Test only Naive kernel with video"
    echo ""
}

# Function to map kernel names to application modes
map_kernel_to_mode() {
    local kernel=$1
    case $kernel in
        naive)
            echo "interactive"
            ;;
        multistream)
            echo "interactive"
            ;;
        cub)
            echo "interactive"
            ;;
        brentkunng)
            echo "interactive"
            ;;
        thrust)
            echo "interactive"
            ;;
        test)
            echo "test"
            ;;
        *)
            echo "Error: Invalid kernel '$kernel'"
            echo "Valid kernels: naive, multistream, cub, brentkunng, thrust, test"
            exit 1
            ;;
    esac
}

# Function to build the project
build_project() {
    echo "Building project..."
    make clean && make build
    
    if [ $? -ne 0 ]; then
        echo "Build failed! Please check your code and dependencies."
        exit 1
    fi
    
    echo "Build successful!"
    echo ""
}

# Parse command line arguments
BUILD_FLAG=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -b|--build)
            BUILD_FLAG=true
            shift
            ;;
        -*)
            echo "Unknown option $1"
            show_usage
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

# Build if requested
if [ "$BUILD_FLAG" = true ]; then
    build_project
fi

# Check if we have the required arguments
if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments"
    echo ""
    show_usage
    exit 1
fi

# Always build the project for actual execution (not just help)
build_project

# Parse arguments - handle both old and new format
VIDEO_SOURCE=$1

# Check if second argument is "singletest" (new 3-argument format)
if [ "$2" = "singletest" ]; then
    MODE="singletest"
    KERNEL=$3
    
    if [ -z "$KERNEL" ]; then
        echo "Error: Kernel name required for singletest mode"
        echo "Usage: $0 [video_source] singletest [kernel]"
        echo "Example: $0 0 singletest thrust"
        exit 1
    fi
else
    # Old 2-argument format: map kernel to mode
    KERNEL=$2
    MODE=$(map_kernel_to_mode "$KERNEL")
fi

echo "Video source: $VIDEO_SOURCE"
echo "Mode: $MODE"
echo "Kernel: $KERNEL"
echo ""

# Validate video source
if [ "$VIDEO_SOURCE" != "0" ] && [ ! -f "$VIDEO_SOURCE" ]; then
    # Try to find file in data directory
    if [ -f "$DATA_DIR/$VIDEO_SOURCE" ]; then
        VIDEO_SOURCE="$DATA_DIR/$VIDEO_SOURCE"
        echo "Found video file: $VIDEO_SOURCE"
    else
        echo "Error: Video source '$VIDEO_SOURCE' not found."
        echo "For webcam, use '0'"
        echo "For video files, ensure the file exists or use full path"
        exit 1
    fi
fi

# Run the application
if [ "$MODE" = "singletest" ]; then
    echo "Starting single kernel test with $KERNEL kernel..."
else
    echo "Starting application with $KERNEL kernel..."
fi
./bin/bluring_part_video.exe "$VIDEO_SOURCE" "$MODE" "$KERNEL"

echo ""
echo "Project completed."