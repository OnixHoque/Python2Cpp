#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define N 10000000
#define MAX_ERR 1e-6


__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

void cuda_sample(int pa, int pb){
    float *a, *b, *out;
    float * d_a, * d_b, *d_out;


    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }

	cudaMalloc((void **)&d_a, sizeof(float) * N);
	cudaMalloc((void **)&d_b, sizeof(float) * N);
	cudaMalloc((void **)&d_out, sizeof(float) * N);
	

    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);


    // Main function
    vector_add<<<pa, pb>>>(d_out, d_a, d_b, N);
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

	printf("%f", out[0]);
cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);
}
