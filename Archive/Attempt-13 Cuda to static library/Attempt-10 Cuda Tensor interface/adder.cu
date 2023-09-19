#include <iostream>
#include <cuda_runtime.h>

// Kernel function to add two arrays element-wise
__global__ void addArrays(float* a, float* b, float* result, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        result[index] = a[index] + b[index];
    }
}

// Function to add two arrays on the GPU
void addArraysOnGPU(float* hostArrayA, float* hostArrayB, float* hostResult, int arraySize) {

    // std::cout << "Array A" << std::endl;

    // for (int i = 0; i < arraySize; ++i) {
    //     std::cout << hostArrayA[i] << " ";
    // }

    // Device arrays
    float* deviceArrayA;
    float* deviceArrayB;
    float* deviceResult;

    // Allocate memory on the GPU
    cudaMalloc((void**)&deviceArrayA, arraySize * sizeof(float));
    cudaMalloc((void**)&deviceArrayB, arraySize * sizeof(float));
    cudaMalloc((void**)&deviceResult, arraySize * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(deviceArrayA, hostArrayA, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceArrayB, hostArrayB, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // Launch the kernel
    addArrays<<<gridSize, blockSize>>>(deviceArrayA, deviceArrayB, deviceResult, arraySize);

    // Copy the result back to the host
    cudaMemcpy(hostResult, deviceResult, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(deviceArrayA);
    cudaFree(deviceArrayB);
    cudaFree(deviceResult);
}
