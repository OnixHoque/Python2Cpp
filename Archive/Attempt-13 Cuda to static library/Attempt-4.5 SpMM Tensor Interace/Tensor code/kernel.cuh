#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include <torch/extension.h>

// Declaration of the CUDA kernel for FusedMM
void fusedmm_cuda(torch::Tensor rowptr, torch::Tensor col,
          torch::optional<torch::Tensor> optional_value, torch::Tensor mat, torch::Tensor out);

#endif  // KERNEL_CUH_