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
TEST_DIR="tests-cuda"

nvcc -O3 -shared -std=c++17 \
    src/fourier.cu \
    -I src/ \
    $TEST_DIR/fourier-bindings.cpp \
    -Xcompiler "-fPIC -Wall -Werror -Wextra -O3" \
    $(python -m pybind11 --includes) \
    -o $TEST_DIR/$MODULE_NAME$(python3-config --extension-suffix) \
    -lcudart

# Running the test Python file
pytest tests-cuda/