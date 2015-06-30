#!/usr/bin/env bash
cd src/main/resources/kernels
rm *.cubin
nvcc -cubin -arch sm_30 *.cu
cd ../../../..