import myadder as adr
import torch
import numpy as np
import time

print("Running test function...")
adr.test_adder()
print("OK. Now passing arguemnt from Python..")

a = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
b = torch.tensor([10, 20, 30, 40, 50], dtype=torch.float32)

a = a.numpy()
b = b.numpy()

c = np.zeros(a.shape[0])

N = a.shape[0]

adr.addArraysOnGPU(a, b, c, N)


