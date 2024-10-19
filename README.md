# CUDA-based Real-Time Video Stream Filtering

## Overview

This project leverages **CUDA** technology to apply real-time blurring to specified regions of video frames. The primary objective is to demonstrate how **GPU acceleration** can be used for video processing tasks like selective blurring, useful in various scenarios such as privacy masking or artistic effects.

Developed as part of a final assignment of Coursera course, this project showcases the efficiency of **GPU computing** in handling large-scale data operations, particularly for time-sensitive applications like real-time video editing.

## Code Organization

- **`data/`**: Stores example video files and datasets used for testing and demonstration purposes. This folder is ignored by Git to keep the repository lightweight. Populate this folder with your own video files for local testing.

- **`src/`**: Stores source code of the project.

- **`lib/`**: Stores header files of the project.

- **`bin/`**: Stores binary files of the project.

- **`README.md`**: Contains the project description and usage instructions.

- **`Makefile`**: Used to compile and build the project. Run `make` in the project directory to compile the source code.

- **`run.sh`**: A script to execute the application with the appropriate input after compilation. Ensure that the `data/` folder contains video files before running this script.

## Dependencies

- **CUDA Toolkit**
- **OpenCV**
- **Omp**

## Prerequisites

Before building and running the project, ensure the following are installed on your system:

- **CUDA Toolkit**: Download and install from [NVIDIA's CUDA Toolkit page](https://developer.nvidia.com/cuda-toolkit) based on your platform.
- Other dependencies: Refer to the Dependencies section.

## Build and Run Instructions

### Windows (using WSL)

To compile and run the project on Windows, use **Windows Subsystem for Linux (WSL)**:

1. Ensure WSL is installed and configured.
2. Follow the Instructions to run on Linux.

### Linux

To build the project on Linux:

1. Navigate to the project directory.
2. Run the following commands:
   ```bash
   $ cd <sample_dir>
   $ chmod +x run.sh
   $ ./run.sh
   ```

## Running the Program

Once the project is compiled and the `run.sh` script is executed:

1. You will see a list of available input files from the `data/` directory:
   ```
   Available input files:
   impp.mp4  input.mp4
   Enter the name of the input file:
   ```

2. Type the name of the desired input file and the program will begin processing the video.

## Results

This is the [link](https://drive.google.com/file/d/1XYbSm9aQYwUQ4VUQVN0f-lHW_9AHIz2c/view?usp=sharing) to the video recording of my project. As you can see in the video, the blur effect is applied in real-time, allowing me to blur some scenes of violent fight scenes. However, it's also easy to notice that the video is still not very smooth and has some lag in certain areas.

The main reason for this issue lies in the CPU's slow reading of image data from the video. Previously, I processed each frame of the video individually, which resulted in the video not stuttering, but the playback time was three times slower than the original video duration. The primary cause of this was the CPU's data reading speed, coupled with the data transfer rate to the GPU being significantly slower than the image processing speed on the GPU. To address this, I used the OpenMP library to leverage the multi-core capabilities of the CPU, along with dynamic parallelism techniques, allowing me to transfer data for 16 frames from the CPU to the GPU and process them simultaneously. As a result, the playback speed of the edited video is now nearly on par with the original video speed. However, the trade-off is that the video will briefly stutter after a fixed period of time. This happens when the CPU has to read the data for 16 frames at once and transfer it to the GPU. Since the GPU's image processing speed is over 2000 times faster than the CPU's reading speed, this stutter is quite noticeable. Overall, the program has been optimized so that the processed video approaches the smoothness of the original video.

