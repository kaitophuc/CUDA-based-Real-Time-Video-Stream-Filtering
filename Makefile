# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv4)

all: clean build run

build:
	$(CXX) -Xcompiler -fopenmp bluring_part_video.cu -rdc=true --std c++20 `pkg-config opencv4 --cflags --libs` -o bluring_part_video.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda

run:
	./bluring_part_video.exe $(ARGS)

clean:
	rm -f *.exe