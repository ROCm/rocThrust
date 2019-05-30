#!/bin/bash

if [ -d docBin ]; then
    rm -rf docBin
fi

#sed -e 's/ROCFFT_EXPORT //g' ../library/include/rocfft.h > rocfft.h
../doxygen doc/thrust.dox

cd source
make clean
make html
cd ..

#rm rocfft.h

