#include <stdio.h>

__global__ void mykernel(){
  printf("hello\n");
}

void gpu_fw(){
  mykernel<<<1,1>>>();
  cudaDeviceSynchronize();
}
