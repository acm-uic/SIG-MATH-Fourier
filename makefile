SRC_DIR=src
BIN_DIR=bin

fourier: $(SRC_DIR)/fourier.cpp
	g++ $< -O3 -flto -o $(BIN_DIR)/fourier