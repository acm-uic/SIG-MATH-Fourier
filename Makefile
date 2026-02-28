# Basic Makefile script for compiling basic code

SRC_DIR=src
BIN_DIR=bin

# Basic single-threaded fourier program compilation and running
basic: $(SRC_DIR)/main.cpp
	mkdir -p $(BIN_DIR)
	g++ $< -O3 -flto -march=native -Wall -Werror -o $(BIN_DIR)/fourier

run_basic: basic
	$(BIN_DIR)/fourier

# CUDA binary compilation and running
cuda: $(SRC_DIR)/main.cu
	mkdir -p $(BIN_DIR)
	nvcc -O3 -dlto -Xcompiler "-Wall -Werror -O3" -arch=native $< -o $(BIN_DIR)/fourier-cuda

run_cuda: cuda
	$(BIN_DIR)/fourier-cuda

# Clean
clean:
	rm -rf $(BIN_DIR)