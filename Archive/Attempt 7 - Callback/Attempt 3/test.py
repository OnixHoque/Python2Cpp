from ctypes import CFUNCTYPE
from ctypes import c_void_p
from ctypes import POINTER
from ctypes import c_int
from ctypes import cdll

lib = cdll.LoadLibrary('./libdiv.so')

FUNC_CAST = CFUNCTYPE(c_int, c_int, c_int)

def op1(x, y):
  return x + y

def op2(x, y):
  return x - y

def op3(x, y):
  return x * y

def op4(x, y):
  return x // y

lib.perform_op(FUNC_CAST(op1))
lib.perform_op(FUNC_CAST(op2))
lib.perform_op(FUNC_CAST(op3))
lib.perform_op(FUNC_CAST(op4))
