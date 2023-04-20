#!/bin/bash
nvcc main.cpp kernel.cu -o matmul
./matmul
