#!/bin/bash

# This is meant to be run from directory root
# i.e. run it like `./python-modules/make-sigMathFourier.sh` from SIG-MATH-Fourier/ directory

# Names and locations
PY_MODULE_NAME="sigMathFourier"
PY_MODULE_DIR="python-modules"
SRC_DIR="src/basic"

# Setting Python environments
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r $PY_MODULE_DIR/requirements.txt

# Binding the C++ to Python
g++ -O3 -Wall -shared -std=c++23 -fPIC \
    $(python -m pybind11 --includes) \
    -I $SRC_DIR/ \
    $PY_MODULE_DIR/$PY_MODULE_NAME.cpp \
    -o $PY_MODULE_DIR/$PY_MODULE_NAME$(python3-config --extension-suffix)