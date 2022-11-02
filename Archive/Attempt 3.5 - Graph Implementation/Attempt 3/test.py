import ctypes
import numpy
import glob
from ctypes import *
# find the shared library, the path depends on the platform and Python version
libfile = glob.glob('build/*/mygraphlib*.so')[0]

# 1. open the shared library
lib = ctypes.CDLL(libfile)


lib.Array_new.argtypes = [ctypes.c_int]
lib.Array_new.restypes = POINTER(c_int)

lib.Array_print.argtypes = [POINTER(c_int), ctypes.c_int]
lib.Array_print.restypes = ctypes.c_void_p

lib.Array_set.argtypes = [ctypes.c_int, ctypes.c_int]
lib.Array_set.restypes = ctypes.c_void_p

lib.Array_destroy.argtypes = [ctypes.c_int, ctypes.c_int]
lib.Array_destroy.restypes = ctypes.c_void_p


p1 = lib.Array_new(10)
print("Array Created!")
lib.Array_print(p1, 10)

