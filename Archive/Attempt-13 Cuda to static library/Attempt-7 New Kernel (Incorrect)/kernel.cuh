#pragma once


// Declaration of the CUDA kernel for FusedMM
//void fusedmm_cuda(int m, int n, int k, int nnz, 
//		const int64_t* indx, const int64_t* ptrb, const float* val,
//		const float* b, float* c, int64_t* prefix_sum, int* nnz_row);

void fusedmm_cuda_coo(int m, int n, int k, int nnz,
                const int64_t* indx, const int64_t* ptrb, const float* val,
                const float* b, float* c);
