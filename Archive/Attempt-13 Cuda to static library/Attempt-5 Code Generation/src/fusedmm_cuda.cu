#include "kernels"

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
