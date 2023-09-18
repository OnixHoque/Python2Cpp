#include "kernel.cuh"
#include <stdio.h>

__global__ void fusedmm_kernel(int m, int n, int k, int nnz, const int64_t* indx, const int64_t* ptrb, const float* val, const float* b, float* c, int nnz_per_block) {
    
    int nnz_start = blockIdx.x * nnz_per_block;
    int nnz_end = min(nnz_start + nnz_per_block, nnz);
    
    for (int i = threadIdx.x; i < m * k; i += blockDim.x) {
        c[i] = 0.0f;
    }
    __syncthreads();

    for (int i = nnz_start; i < nnz_end; i++) {
        
	int row = ptrb[i];
        int col = indx[i];
        float val_ij = val[i];

        float temp = 0.0f;
       	
        // blockDim.x -- Total threads in the block.	
	for (int j = threadIdx.x; j < k; j += blockDim.x) {
	    temp += val_ij * b[col * k + j];
        }
        atomicAdd(&c[row * k + threadIdx.x], temp);
    }
}

void fusedmm_cuda_coo(int m, int n, int k, int nnz,
                               const int64_t* indx, const int64_t* ptrb, const float* val,
                               const float* b, float* c) {

	int nnz_per_block = 1024;
    	int threads_per_block = 128;
	int num_blocks = (nnz + nnz_per_block - 1) / nnz_per_block;
    	
	printf("Num Blocks: %d\n", num_blocks);
	
	fusedmm_kernel<<<num_blocks, threads_per_block>>>(m, n, k, nnz, indx, ptrb, val, b, c, nnz_per_block);

	cudaDeviceSynchronize();
}