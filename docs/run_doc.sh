#!/bin/bash

if [ -d docBin ]; then
    rm -rf docBin
fi

doxygen thrust.dox

make clean
make html
