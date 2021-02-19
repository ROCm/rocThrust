#!/bin/bash

set -eu

# Make this directory the PWD
cd "$(dirname "${BASH_SOURCE[0]}")"

# Build doxygen info
rm -rf docBin
doxygen Doxyfile
