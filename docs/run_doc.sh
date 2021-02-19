#!/bin/bash

set -eu

# Make this directory the PWD
cd "$(dirname "${BASH_SOURCE[0]}")"

# Build doxygen info
./run_doxygen.sh

# Build sphinx docs
make clean
make html
make latexpdf
