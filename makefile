SRC_DIR=src
BIN_DIR=bin

fourier: $(SRC_DIR)/fourier.cpp
	g++ $< -o $(BIN_DIR)/fourier