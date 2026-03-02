#!/bin/bash

# This is meant to be run from directory root
# i.e. run it like `./python-modules/make-acmFourierCUDA.sh` from SIG-MATH-Fourier/ directory

# Names and locations
PY_MODULE_NAME="acmFourierCUDA"
PY_MODULE_DIR="python-PY_MODULEs"

# Setting Python environments
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r $PY_MODULE_DIR/requirements.txt

# Binding the C++ to Python
nvcc -O3 -shared -std=c++20 \
    src/fourier.cu \
    -I src/ \
    $PY_MODULE_DIR/$PY_MODULE_NAME.cpp \
    -Xcompiler "-fPIC -Wall -Werror -Wextra -O3" \
    $(python -m pybind11 --includes) \
    -o $PY_MODULE_DIR/$PY_MODULE_NAME$(python3-config --extension-suffix) \
    -lcudart