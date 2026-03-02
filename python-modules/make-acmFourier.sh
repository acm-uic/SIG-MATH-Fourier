#!/bin/bash

# This is meant to be run from directory root
# i.e. run it like `./python-modules/make-acmFourier.sh` from SIG-MATH-Fourier/ directory

# Names and locations
MODULE_NAME="acmFourier"
MODULE_DIR="python-modules"

# Setting Python environments
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r $MODULE_DIR/requirements.txt

# Binding the C++ to Python
g++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python -m pybind11 --includes) \
    -I src/ \
    $MODULE_DIR/$MODULE_NAME.cpp \
    -o $MODULE_DIR/$MODULE_NAME$(python3-config --extension-suffix)