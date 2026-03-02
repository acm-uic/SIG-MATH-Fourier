#!/bin/bash

# This is meant to be run from directory root
# i.e. run it like `./tests/run-tests.sh` from SIG-MATH-Fourier/ directory


# Setting Python environments
python3 -m venv .venv
source .venv/bin/activate
#pip install --upgrade pip
#pip install -r tests/requirements.txt

# Binding the C++ to Python
MODULE_NAME="acmFourierCUDA"

nvcc src/fourier.cu \
    -I src/ \
    tests/fourier-bindings.cpp \
    -O3 -shared -std=c++20 \
    -Xcompiler "-fPIC -Wall -Werror -O3 -std=c++23" \
    $(python -m pybind11 --includes) \
    -o tests/$MODULE_NAME$(python3-config --extension-suffix)

# Running the test Python file
pytest tests-cuda/