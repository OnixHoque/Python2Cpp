#include "kernel.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// __global__ void fusedmm_kernel(int m, int n, int k, int nnz, const int* ptrb, const int* indx, const float* val, const float** b, float** c) {

//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < m * k) {
//         int row = tid % m;
//         int col = tid / m;
//         float res = 0.0f;
//         for (int i = ptrb[row]; i < ptrb[row + 1]; ++i) {
//             int ind = indx[i]; //col indices array
//             res += b[ind][col] * val[i];
//         }
//         c[row][col] = res;
//     }
// }

__global__ void fusedmm_kernel(int m, int n, int k, int nnz, 
                               const int* indx, const int* ptrb, const float* val, 
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

void fusedmm_cuda(torch::Tensor rowptr, torch::Tensor col,
          torch::optional<torch::Tensor> optional_value, torch::Tensor mat, torch::Tensor out) {

  // Basic Memory Checks: evaluates if data is present in GPU Memory
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  if (optional_value.has_value())
    CHECK_CUDA(optional_value.value());
  CHECK_CUDA(mat);
  cudaSetDevice(rowptr.get_device());

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  if (optional_value.has_value()) {
    CHECK_INPUT(optional_value.value().dim() == 1);
    CHECK_INPUT(optional_value.value().size(0) == col.size(0));
  }
  CHECK_INPUT(mat.dim() >= 2);

  mat = mat.contiguous();

  int m = rowptr.numel() - 1;
  int n = mat.size(-2);
  int k = mat.size(-1);
  int nnz = optional_value.value().size(0);
  int rows = m;
  int cols = n;
  int64_t *ptrb = rowptr.data_ptr<int64_t>();
  int64_t *indx = col.data_ptr<int64_t>();
  float *val = value.value().data_ptr<float>();
  int64_t *ptre = ptrb + 1;
  float *b = mat.data_ptr<float>();
  float *c = out.data_ptr<float>();

  // Compute the number of threads per block and blocks per grid
  int row_per_block = 128/k;
  int n_block = (m+row_per_block-1)/row_per_block;

  // Launch the CUDA kernel
  fusedmm_kernel<<<dim3(n_block,1,1),dim3(k, row_per_block, 1)>>>(m, n, k, nnz, indx, ptrb, val, b, c)
}