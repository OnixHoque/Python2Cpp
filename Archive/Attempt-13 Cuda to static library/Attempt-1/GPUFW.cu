#include <stdio.h>

__global__ void mykernel() {
	printf("Hello!");
}

void gpu_fw() {
	mykernel<<<1, 1>>>();
	cudaDeviceSynchronize();
}
