blc_per_row = 256

code_template = r'''
#pragma once
#include <cuda_runtime.h>

/*
__global__ void fusedmm_kernel(int m, int n, int k, int nnz, const int64_t* indx, const int64_t* ptrb, const float* val, const float* b, float* c) {

     int tid = blockIdx.x * blockDim.x + threadIdx.x;
     if (tid < m * k) {
         int row = tid % m;
         int col = tid / m;
         float res = 0.0f;
         for (int i = ptrb[row]; i < ptrb[row + 1]; ++i) {
             int ind = indx[i]; //col indices array    
	 	res += b[ind + col * n] * val[i];
	 }
     c[row + col * m] = res;
     }
}
*/

__global__ void fusedmm_kernel(int m, int n, int k, int nnz, 
                               const int64_t* indx, const int64_t* ptrb, const float* val, 
                               const float* b, float* c)
{
    int rid = blockDim.y*blockIdx.x+threadIdx.y;
    if (rid<m) {
      int cid = (blockIdx.y<<5)+threadIdx.x;
      int lb = ptrb[rid];
      int hb = ptrb[(rid+1)];
      int offset = 0;
      float acc=0;
      if (blockIdx.y!=gridDim.y-1){
          for (int ptr = lb; ptr<hb; ptr++) {
              offset = indx[ptr]*k+cid;
              acc += val[ptr]*b[offset];
          }
          c[(rid*k+cid)] = acc;
      }
      else {
          for (int ptr = lb; ptr<hb; ptr++) {
              if (cid<k) {
                offset = indx[ptr]*k+cid;
              }
              acc += val[ptr]*b[offset];
          }
          if (cid<k) {
            c[(rid*k+cid)] = acc;
          }
      }
    }
}

void fusedmm_cuda_bl{{blc_per_row}}(int m, int n, int k, int nnz,
                               const int64_t* indx, const int64_t* ptrb, const float* val,
                               const float* b, float* c) {

  // Compute the number of threads per block and blocks per grid
  int row_per_block = {blc_per_row}/k;
  int n_block = (m+row_per_block-1)/row_per_block;

  // Launch the CUDA kernel
  fusedmm_kernel<<<dim3(n_block,1,1),dim3(k, row_per_block, 1)>>>(m, n, k, nnz, indx, ptrb, val, b, c);
}

'''

header_template = r'''void fusedmm_cuda_bl{{blc_per_row}}(int m, int n, int k, int nnz, const int64_t* indx, const int64_t* ptrb, const float* val, const float* b, float* c);'''

headers = []
for k in range(128, 513, 8):
    with open(f'generated/spmm_kernel_bl{k}.cu', 'w') as f:
        f.write(code_template.replace('{{blc_per_row}}', k))
        headers += [header_template.replace('{{blc_per_row}}', k)]
    
    with open('kernel.cuh', 'w') as f:
        f.write('\n'.join(headers))