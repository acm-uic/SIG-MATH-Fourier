#!/bin/bash

export OMP_NUM_THREADS=$(nproc)
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

g++ src/main.cpp -O3 -flto -fopenmp -march=native -Wall -Werror -o bin/fourier
./bin/fourier