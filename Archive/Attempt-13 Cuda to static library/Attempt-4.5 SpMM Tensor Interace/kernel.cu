// #include "kernel.cuh"
#include <iostream>
#include <cuda_runtime.h>

#define INDEXTYPE int64_t
#define VALUETYPE float

__global__ void fusedmm_spmm_trusted_kernel(int m, int n, int k, int nnz, 
                               const int64_t* indx, const int64_t* ptrb, const float* val, 
                               const float* b, float* c)
{
    int rid = blockDim.y*blockIdx.x+threadIdx.y;
    if (rid<m) {
      int cid = (blockIdx.y<<5)+threadIdx.x;
      // int cid = (blockIdx.y * k)+threadIdx.x;
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
/*void fusedmm_cuda(int m, int n, int k, int nnz,
                               const int64_t* indx, const int64_t* ptrb, const float* val,
                               const float* b, float* c) */
void fusedmm_cuda
(
   const char tkern,       // kernel variations
   const INDEXTYPE m,      // rows of A 
   const INDEXTYPE n,      // rows of B
   const INDEXTYPE k,      // dimension: col of A and B
   const VALUETYPE alpha,  // not used yet  
   const INDEXTYPE nnz,    // nonzeros  
   const INDEXTYPE rows,   // number of rows for sparse matrix 
   const INDEXTYPE cols,   // number of columns for sparse matrix 
   const VALUETYPE *val,   // NNZ value  
   const INDEXTYPE *indx,  // colids -> column indices 
   const INDEXTYPE *pntrb, // starting index for rowptr
   const INDEXTYPE *pntre, // ending index for rowptr
   const VALUETYPE *a,     // Dense A (X) matrix
   const INDEXTYPE lda,    // leading dimension of A (col size since A row-major)  
   const VALUETYPE *b,     // Dense B matrix
   const INDEXTYPE ldb,    // leading dimension of B (col size since B row-major)  
   const VALUETYPE beta,   // beta value 
   VALUETYPE *c,           // Dense matrix c
   const INDEXTYPE ldc     // leading dimension of c (col size since C row-major) 
)
{

  // Compute the number of threads per block and blocks per grid
  int row_per_block = 256/k;
  int n_block = (m+row_per_block-1)/row_per_block;

  // Launch the CUDA kernel
  fusedmm_spmm_trusted_kernel<<<dim3(n_block,1,1),dim3(k, row_per_block, 1)>>>(m, n, k, nnz, indx, pntrb, val, b, c);
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
    fusedmm_cuda('m', m, n, k, 1, nnz, 0, 0,val_device, indx_device, ptrb_device,ptrb_device+1 , 0, 0,  mat_device, 0, 0, out_device, 0);
	
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
