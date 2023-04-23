#include "generated/include/spmm_header.cuh"

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