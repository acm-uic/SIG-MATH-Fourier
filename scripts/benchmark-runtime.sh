#!/bin/bash

LOG_DIR=logs/
mkdir -p logs

echo -e "For 2^16 points 1D Fourier Transforms\n " > $LOG_DIR/runtime.log

echo "Basic runtimes: " >> $LOG_DIR/runtime.log
make -s run_basic >> $LOG_DIR/runtime.log 

echo "" >> $LOG_DIR/runtime.log
echo "CUDA runtimes: " >> $LOG_DIR/runtime.log
make -s run_cuda >> $LOG_DIR/runtime.log


cat $LOG_DIR/runtime.log