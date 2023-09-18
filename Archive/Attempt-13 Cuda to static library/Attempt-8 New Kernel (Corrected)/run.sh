#!/bin/bash
nvcc main.cpp kernel_coo.cu -o matmul
./matmul > out.txt
