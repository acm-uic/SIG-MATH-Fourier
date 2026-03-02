#!/bin/bash

# This is meant to be run from directory root
# i.e. run it like `./tests-cuda/run-tests.sh` from SIG-MATH-Fourier/ directory

# Names of stuff
MODULE_NAME="acmFourierCUDA"
MODULE_DIR="python-modules"
TEST_DIR="tests-cuda"

# Compiling the basic C++ code into a Python module
source python-modules/make-$MODULE_NAME.sh

# Move the compiled Python module here
mv $MODULE_DIR/$MODULE_NAME$(python3-config --extension-suffix) $TEST_DIR

# Running the test Python files
pytest $TEST_DIR