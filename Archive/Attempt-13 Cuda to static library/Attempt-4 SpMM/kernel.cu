#include "kernel.cuh"
#include <iostream>
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
void fusedmm_cuda(int m, int n, int k, int nnz,
                               const int64_t* indx, const int64_t* ptrb, const float* val,
                               const float* b, float* c) {

  // Compute the number of threads per block and blocks per grid
  int row_per_block = 256/k;
  int n_block = (m+row_per_block-1)/row_per_block;

  // Launch the CUDA kernel
  fusedmm_kernel<<<dim3(n_block,1,1),dim3(k, row_per_block, 1)>>>(m, n, k, nnz, indx, ptrb, val, b, c);
}


void cuda_spmm_test()
{

    // Allocate host (CPU) memory
    int64_t *ptrb = new int64_t[5]{0, 1, 2, 3, 4};
    int64_t *indx = new int64_t[4]{0, 1, 2, 3}; 
    float *val = new float[4]{1.0, 2.0, 3.0, 4.0};

    float mat[4][4] = {
	    		{2.0, 2.0, 2.0, 2.0},
	    		{2.0, 2.0, 2.0, 2.0},
	    		{2.0, 2.0, 2.0, 2.0},
	    		{2.0, 2.0, 2.0, 2.0}};
    float out[4][4] = {
                        {0, 0, 0, 0},
                        {0, 0, 0, 0},
                        {0, 0, 0, 0},
                        {0, 0, 0, 0}};
    int m=4;
    int n=4;
    int k=4;
    int nnz=4;
    int rows=m;
    int cols=n;

    // Allocate device (GPU) memory
    int64_t *ptrb_device, *indx_device; 
    float *val_device, *mat_device, *out_device;
    cudaMalloc(&ptrb_device, (m+1) * sizeof(int64_t));
    cudaMalloc(&indx_device, nnz * sizeof(int64_t));
    cudaMalloc(&val_device, nnz * sizeof(float));
    cudaMalloc(&mat_device, n * k * sizeof(float));
    cudaMalloc(&out_device, m * k * sizeof(float)); 
    
    // Copy input data from host to device
    cudaMemcpy(ptrb_device, ptrb, (m+1) * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(indx_device, indx, nnz * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(val_device, val, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mat_device, mat, n * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_device, out, m * k * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    fusedmm_cuda(m, n, k, nnz, indx_device, ptrb_device, val_device, mat_device, out_device);
	
    // Copy output data from device to host
    cudaMemcpy(out, out_device, m * k * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print output
    for (int i = 0; i < m; i++) {
	    for(int j=0; j<k; j++){
        	std::cout << out[i][j] << " ";
	    }
	    std::cout << "\n";
    }
    std::cout << std::endl;

    // Free memory
    cudaFree(ptrb_device);
    cudaFree(indx_device);
    cudaFree(val_device);
    cudaFree(mat_device);
    cudaFree(out_device);

    //return 0;
}
