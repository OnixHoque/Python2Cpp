import ctypes
import numpy
import glob

# find the shared library, the path depends on the platform and Python version
libfile = glob.glob('build/*/mysum*.so')[0]

# 1. open the shared library
mylib = ctypes.CDLL(libfile)

# 2. tell Python the argument and result types of function mysum
mylib.mysum.restype = ctypes.c_longlong
mylib.mysum.argtypes = [ctypes.c_int, 
                        numpy.ctypeslib.ndpointer(dtype=numpy.int32)]

array = numpy.arange(0, 10, 1, numpy.int32)
print("Sum of array ", array)

# 3. call function mysum
array_sum = mylib.mysum(len(array), array)

print('sum of array: {}'.format(array_sum))
