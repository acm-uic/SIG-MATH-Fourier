# Basic Makefile script for compiling basic code

SRC_DIR=src
BIN_DIR=bin

basic: $(SRC_DIR)/main.cpp
	mkdir -p $(BIN_DIR)
	g++ $< -O3 -flto -march=native \
		-Wall -Werror \
		-o $(BIN_DIR)/fourier

run_basic: basic
	$(BIN_DIR)/fourier

clean:
	rm -rf bin/