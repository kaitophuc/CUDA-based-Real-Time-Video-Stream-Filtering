#!/bin/bash

# CUDA-based Real-Time Video Stream Filtering
# Comprehensive execution script with kernel testing capabilities

echo "=== CUDA-based Real-Time Video Stream Filtering ==="
echo ""

# Directory containing input files
DATA_DIR="./data"

# Function to display usage information
show_usage() {
    echo "Usage: $0 [OPTIONS] [video_source] [mode]"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help     Show this help message"
    echo "  -b, --build    Build the project before running"
    echo "  -i, --interactive  Interactive mode selection"
    echo ""
    echo "Video sources:"
    echo "  0              Use webcam"
    echo "  video.mp4      Use video file from data/ directory"
    echo "  full/path.mp4  Use video file with full path"
    echo ""
    echo "Modes:"
    echo "  test           Test all kernels and show performance comparison"
    echo "  benchmark      Test all kernels then run best one interactively"
    echo "  interactive    Choose kernel and run interactively (default)"
    echo "  webcam_benchmark  Test all kernels with webcam (10s each)"
    echo ""
    echo "Examples:"
    echo "  $0                           # Interactive mode selection"
    echo "  $0 --build 0 webcam_benchmark # Build then test kernels with webcam"
    echo "  $0 0 test                    # Test all kernels with webcam"
    echo "  $0 data/input.mp4 benchmark  # Benchmark with video file"
    echo "  $0 0 interactive             # Interactive mode with webcam"
    echo ""
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

# Function to check if binary exists
check_binary() {
    if [ ! -f "./bin/bluring_part_video.exe" ]; then
        echo "Binary not found. Building project..."
        build_project
    fi
}

# Function to interactive mode selection
interactive_selection() {
    echo "Choose execution mode:"
    echo "1. Live Camera (Interactive)"
    echo "2. Live Camera (Kernel Testing)"
    echo "3. Live Camera (Webcam Benchmark)"
    echo "4. Recorded Video (Interactive)"
    echo "5. Recorded Video (Kernel Testing)"
    echo "6. Exit"
    echo ""
    read -p "Enter your choice (1-6): " CHOICE
    
    case $CHOICE in
        1)
            echo "Running interactive mode with live camera..."
            ./bin/bluring_part_video.exe 0 interactive
            ;;
        2)
            echo "Testing all kernels with live camera..."
            ./bin/bluring_part_video.exe 0 test
            ;;
        3)
            echo "Running webcam benchmark (10 seconds per kernel)..."
            ./bin/bluring_part_video.exe 0 webcam_benchmark
            ;;
        4)
            select_video_file "interactive"
            ;;
        5)
            select_video_file "test"
            ;;
        6)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid choice. Please enter 1-6."
            exit 1
            ;;
    esac
}

# Function to select video file
select_video_file() {
    local mode=$1
    
    # Check if data directory exists
    if [ ! -d "$DATA_DIR" ]; then
        echo "Data directory $DATA_DIR does not exist."
        echo "Please create the directory and add video files, or use webcam mode."
        exit 1
    fi
    
    # List available input files
    echo "Available input files in $DATA_DIR:"
    if [ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
        echo "No files found in $DATA_DIR directory."
        echo "Please add video files to the data directory."
        exit 1
    fi
    
    ls "$DATA_DIR"
    echo ""
    
    # Prompt user to select an input file
    read -p "Enter the name of the input file: " INPUT_FILE
    
    # Check if the selected file exists
    if [ ! -f "$DATA_DIR/$INPUT_FILE" ]; then
        echo "Input file $DATA_DIR/$INPUT_FILE does not exist."
        exit 1
    fi
    
    # Run the project with the selected input file
    echo "Running project with input file $DATA_DIR/$INPUT_FILE in $mode mode..."
    ./bin/bluring_part_video.exe "$DATA_DIR/$INPUT_FILE" "$mode"
}

# Parse command line arguments
BUILD_FLAG=false
INTERACTIVE_FLAG=false

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
        -i|--interactive)
            INTERACTIVE_FLAG=true
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

# Check if binary exists
check_binary

# If interactive flag is set or no arguments provided, show interactive menu
if [ "$INTERACTIVE_FLAG" = true ] || [ $# -eq 0 ]; then
    interactive_selection
    echo ""
    echo "Project completed."
    exit 0
fi

# Parse remaining arguments for direct execution
if [ $# -ge 1 ]; then
    VIDEO_SOURCE=$1
    MODE=${2:-interactive}
    
    echo "Video source: $VIDEO_SOURCE"
    echo "Mode: $MODE"
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
    
    # Validate mode
    case $MODE in
        test|benchmark|interactive|webcam_benchmark)
            ;;
        *)
            echo "Error: Invalid mode '$MODE'"
            echo "Valid modes: test, benchmark, interactive, webcam_benchmark"
            exit 1
            ;;
    esac
    
    # Run the application
    echo "Starting application..."
    ./bin/bluring_part_video.exe "$VIDEO_SOURCE" "$MODE"
else
    show_usage
    exit 1
fi

echo ""
echo "Project completed."