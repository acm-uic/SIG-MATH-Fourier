#!/bin/bash

# This is meant to be run from directory root
# i.e. run it like `./python-modules/make-acmFourierCUDA.sh` from SIG-MATH-Fourier/ directory

# Names and locations
MODULE_NAME="acmFourierCUDA"
MODULE_DIR="python-modules"

# Setting Python environments
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r $MODULE_DIR/requirements.txt

# Binding the C++ to Python
nvcc -O3 -shared -std=c++20 \
    src/fourier.cu \
    -I src/ \
    $MODULE_DIR/$MODULE_NAME.cpp \
    -Xcompiler "-fPIC -Wall -Werror -Wextra -O3" \
    $(python -m pybind11 --includes) \
    -o $MODULE_DIR/$MODULE_NAME$(python3-config --extension-suffix) \
    -lcudart