#!/bin/bash

# This is meant to be run from directory root
# i.e. run it like `./python-modules/make-sigMathFourierCUDA.sh` from SIG-MATH-Fourier/ directory

# Names and locations
PY_MODULE_NAME="sigMathFourierCUDA"
PY_MODULE_DIR="python-modules"
SRC_DIR="src/cuda"

# Setting Python environments
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r $PY_MODULE_DIR/requirements.txt

# Binding the C++ to Python
nvcc -O3 -shared -std=c++20 \
    $SRC_DIR/fourier.cu \
    -I $SRC_DIR \
    $PY_MODULE_DIR/$PY_MODULE_NAME.cpp \
    -Xcompiler "-fPIC -Wall -Werror -Wextra -O3 -std=c++20" \
    $(python -m pybind11 --includes) \
    -o $PY_MODULE_DIR/$PY_MODULE_NAME$(python3-config --extension-suffix) \
    -lcudart