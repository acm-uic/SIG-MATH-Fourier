#!/bin/bash

# This is meant to be run from directory root
# i.e. run it like `./tests/run-tests.sh` from SIG-MATH-Fourier/ directory


# Setting Python environments
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r tests/requirements.txt

# Binding the C++ to Python
MODULE_NAME="acmFourier"

g++ -O3 -Wall -Werror -shared -std=c++17 -fPIC -fopenmp \
    $(python -m pybind11 --includes) \
    -I src/ \
    tests/fourier-bindings.cpp \
    -o tests/$MODULE_NAME$(python3-config --extension-suffix)

# Running the test Python file
pytest tests/