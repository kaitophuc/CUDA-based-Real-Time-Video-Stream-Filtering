CXX = nvcc

TARGET = bluring_part_video
SRC_DIR = src
LIB_DIR = lib
BIN_DIR = bin

# Source files for modular approach (automatically detect all .cu files in src/)
SOURCES = $(wildcard $(SRC_DIR)/*.cu)
# Header files for dependency tracking (automatically detect all .hpp files in lib/)
HEADERS = $(wildcard $(LIB_DIR)/*.hpp)

all: clean build run

build: $(HEADERS)
	mkdir -p $(BIN_DIR)
	$(CXX) -Xcompiler -fopenmp -rdc=true --std=c++20 -Wno-deprecated-gpu-targets -I/usr/local/cuda/include -I$(LIB_DIR) `pkg-config opencv4 --cflags --libs` -lcuda -lcudnn $(SOURCES) -o $(BIN_DIR)/$(TARGET).exe

run:
	./$(BIN_DIR)/$(TARGET).exe $(ARGS)

clean:
	rm -f $(BIN_DIR)/*.exe