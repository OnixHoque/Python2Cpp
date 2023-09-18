// #include "kernel.cuh"
#include <stdio.h>

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
    //__shared__  float vop[];
    //vop[tid] = 0;

    int rid = blockDim.y*blockIdx.x+threadIdx.y; // row_id
    if (rid<m) {
      //int cid = (blockIdx.y*k)+threadIdx.x;
      int cid = threadIdx.x; // column_id: 0, 1, 2, 3,  ... k-1
      int lb = ptrb[rid];
      int hb = ptrb[(rid+1)];
      int offset = 0;
      float acc=0;
      for (int ptr = lb; ptr<hb; ptr++) { // CSR: [1, 0, 2, 0]
          if (cid<k) {
            offset = indx[ptr]*k+cid; // 0*k + [0, 1, 2, 3, ... k-1]
          }
	  // VOP + AOP
          acc += val[ptr]*b[offset];
      }
      if (cid<k) {
        c[(rid*k+cid)] = acc;
      }
   }
}

void fusedmm_cuda(int m, int n, int k, int nnz,
                               const int64_t* indx, const int64_t* ptrb, const float* val,
                               const float* b, float* c) {

  // Compute the number of threads per block and blocks per grid
  int row_per_block = 32/k;
  //int row_per_block = 1; // single row is processed by single block 
  int n_block = (m+row_per_block-1)/row_per_block; // m - rows, m - blocks
  printf("n_block: %d row_per_block: %d\n",n_block, row_per_block);
  // Launch the CUDA kernel
  fusedmm_kernel<<<dim3(n_block,1,1),dim3(k, row_per_block, 1)>>>(m, n, k, nnz, indx, ptrb, val, b, c);
}

/*
void fusedmm_cuda(int m, int n, int k, int nnz,
                               const int64_t* indx, const int64_t* ptrb, const float* val,
                               const float* b, float* c, int64_t* prefix_sum, int* nnz_row) {

  // Compute the number of threads per block and blocks per grid
  // int row_per_block = 256/k;
  // int n_block = (m+row_per_block-1)/row_per_block;

  int num_blocks = m;
  int target_nnz_per_block = (nnz + num_blocks - 1) / num_blocks;

  
  // Storing row index to block index mapping
  int row_to_blocks[m];
  int current_block = 0;
  int current_sum = 0;
  for (int i = 0; i < m; i++) {
      int nnz_in_row = nnz_row[i];
      if (current_sum + nnz_in_row > (current_block+1) * target_nnz_per_block) {
          current_block++;
          current_sum = prefix_sum[i];
      }
      row_to_blocks[i] = current_block;
      current_sum = current_sum + nnz_in_row;
  }

  for(int i=0; i<m; i++){
    printf("%d\n", row_to_blocks[i]);
  }

  // Launch the CUDA kernel
  //fusedmm_kernel<<<dim3(n_block,1,1),dim3(k, 1, 1)>>>(m, n, k, nnz, indx, ptrb, val, b, c);
}*/
