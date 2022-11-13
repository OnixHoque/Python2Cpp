import ctypes
import numpy
import glob

# find the shared library, the path depends on the platform and Python version
libfile = glob.glob('./build/*/libmath*.so')[0]

# 1. open the shared library
mylib = ctypes.CDLL(libfile)


mylib.my_add.restype = ctypes.c_int
mylib.my_add.argtypes = [ctypes.c_int, ctypes.c_int]


mylib.my_sub.restype = ctypes.c_int
mylib.my_sub.argtypes = [ctypes.c_int, ctypes.c_int]


add_res = mylib.my_add(20, 10)
print('20 + 10 = {}'.format(add_res))

sub_res = mylib.my_sub(20, 10)
print('20 - 10 = {}'.format(sub_res))
