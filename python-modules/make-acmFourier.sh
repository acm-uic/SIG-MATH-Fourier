#!/bin/bash

# This is meant to be run from directory root
# i.e. run it like `./python-PY_MODULEs/make-acmFourier.sh` from SIG-MATH-Fourier/ directory

# Names and locations
PY_MODULE_NAME="acmFourier"
PY_MODULE_DIR="python-PY_MODULEs"

# Setting Python environments
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r $PY_MODULE_DIR/requirements.txt

# Binding the C++ to Python
g++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python -m pybind11 --includes) \
    -I src/ \
    $PY_MODULE_DIR/$PY_MODULE_NAME.cpp \
    -o $PY_MODULE_DIR/$PY_MODULE_NAME$(python3-config --extension-suffix)