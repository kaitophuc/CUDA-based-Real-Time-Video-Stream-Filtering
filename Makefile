CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv4)
TARGET = bluring_part_video
SRC_DIR = src
LIB_DIR = lib
BIN_DIR = bin

all: clean build run

build:
	mkdir -p $(BIN_DIR)
	$(CXX) -Xcompiler -fopenmp $(SRC_DIR)/$(TARGET).cu -rdc=true --std c++20 `pkg-config opencv4 --cflags --libs` -o $(BIN_DIR)/$(TARGET).exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda -lcudnn

run:
	./$(BIN_DIR)/$(TARGET).exe $(ARGS)

clean:
	rm -f $(BIN_DIR)/*.exe