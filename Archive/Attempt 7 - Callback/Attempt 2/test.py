from ctypes import CFUNCTYPE
from ctypes import c_void_p
from ctypes import POINTER
from ctypes import c_int
from ctypes import cdll
lib = cdll.LoadLibrary('./libdiv.so')
CMPFUNC = CFUNCTYPE(c_void_p, POINTER(c_int), POINTER(c_int))
def py_cmp_func(s, r):
  print (f'Quotient is {s[0]} , remainder is {r[0]}')
cmp_func = CMPFUNC(py_cmp_func)
lib.divide(cmp_func, 3, 5)
