import ctypes
import numpy
import glob

# find the shared library, the path depends on the platform and Python version
libfile = glob.glob('./libmath*.so')[0]

# 1. open the shared library
mylib = ctypes.CDLL(libfile)

# 2. tell Python the argument and result types of function mysum
mylib.my_add.restype = ctypes.c_int
mylib.my_add.argtypes = [ctypes.c_int, ctypes.c_int]

# array = numpy.arange(0, 10, 1, numpy.int32)
# print("Sum of array ", array)

# # 3. call function mysum
array_sum = mylib.my_add(20, 10)

print('sum of array: {}'.format(array_sum))

# print("Done")