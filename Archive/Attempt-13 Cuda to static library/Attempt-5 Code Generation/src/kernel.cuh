// #include "spmm_kernel_K0"

// Declaration of the CUDA kernel for FusedMM
void fusedmm_cuda_bl256(int m, int n, int k, int nnz, const int64_t* indx, const int64_t* ptrb, const float* val, 		const float* b, float* c);
