#define SPMM_KERNEL_COUNT 2
        
    typedef void (*gfusedMM_spmm_t) ( const char transa, const INDEXTYPE m, 
      const INDEXTYPE n, const INDEXTYPE k,const float alpha, const INDEXTYPE nnz, 
      const INDEXTYPE rows, const INDEXTYPE cols, const float *val, const INDEXTYPE *indx, 
      const INDEXTYPE *pntrb, const INDEXTYPE *pntre, const float *A, 
      const INDEXTYPE lda, const float *B, const INDEXTYPE ldb, 
      const float beta, float *C, const INDEXTYPE ldc);void gfusedMM_bl8_spmm_csr (const char transa, const INDEXTYPE m, const INDEXTYPE n, 
      const INDEXTYPE k,const float alpha, const INDEXTYPE nnz, 
      const INDEXTYPE rows, const INDEXTYPE cols, const float *val, const INDEXTYPE *indx, 
      const INDEXTYPE *pntrb, const INDEXTYPE *pntre, const float *A, 
      const INDEXTYPE lda, const float *B, const INDEXTYPE ldb, 
      const float beta, float *C, const INDEXTYPE ldc);

void gfusedMM_bl16_spmm_csr (const char transa, const INDEXTYPE m, const INDEXTYPE n, 
      const INDEXTYPE k,const float alpha, const INDEXTYPE nnz, 
      const INDEXTYPE rows, const INDEXTYPE cols, const float *val, const INDEXTYPE *indx, 
      const INDEXTYPE *pntrb, const INDEXTYPE *pntre, const float *A, 
      const INDEXTYPE lda, const float *B, const INDEXTYPE ldb, 
      const float beta, float *C, const INDEXTYPE ldc);__global__ void gcfusedMM_bl8_spmm_csr (int m, int n, int k, int nnz, 
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

void gfusedMM_bl8_spmm_csr
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
  int row_per_block = 8/k;
  int n_block = (m+row_per_block-1)/row_per_block;

  // Launch the CUDA kernel
  gcfusedMM_bl8_spmm_csr<<<dim3(n_block,1,1),dim3(k, row_per_block, 1)>>>(m, n, k, nnz, indx, pntrb, val, b, c);
}

__global__ void gcfusedMM_bl16_spmm_csr (int m, int n, int k, int nnz, 
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

void gfusedMM_bl16_spmm_csr
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
  int row_per_block = 16/k;
  int n_block = (m+row_per_block-1)/row_per_block;

  // Launch the CUDA kernel
  gcfusedMM_bl16_spmm_csr<<<dim3(n_block,1,1),dim3(k, row_per_block, 1)>>>(m, n, k, nnz, indx, pntrb, val, b, c);
}

gfusedMM_spmm_t gfusedMM_spmm[] = { gfusedMM_bl8_spmm_csr,
gfusedMM_bl16_spmm_csr };