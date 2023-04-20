#include <iostream>
#include <torch/extension.h>
#include <torch/torch.h>
#include "kernel.cuh"

int main() {
  
  // initialize rowptr tensor
  torch::Tensor rowptr = torch::tensor({0, 1, 2, 3, 4}, torch::kInt64);

  // initialize col tensor
  torch::Tensor col = torch::tensor({0, 2, 3, 1}, torch::kInt64);

  // initialize value tensor
  torch::Tensor value = torch::tensor({1.0, 2.0, 3.0, 4.0}, torch::kFloat32);

  // initialize mat tensor
  torch::Tensor mat = torch::randn({4, 8}, torch::kFloat32);

  // initialize out tensor
  torch::Tensor out = torch::zeros({4, 8}, torch::kFloat32);

  // call the function
  fusedmm_cuda(rowptr, col, value, mat, out);

  return 0;
}