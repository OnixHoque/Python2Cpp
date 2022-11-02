import ctypes
import numpy
import glob
from ctypes import *
# find the shared library, the path depends on the platform and Python version
libfile = glob.glob('build/*/mygraphlib*.so')[0]

# 1. open the shared library
lib = ctypes.CDLL(libfile)


class GRAPH_STRUCTURE(Structure):
    _fields_ = [
        ("adjMat", POINTER(c_int)),
        ("N", c_int),
        ("biDirectional", c_int)]

# print("lib loaded!")

# import ctypes
# lib = ctypes.cdll.LoadLibrary('./libfoo.so')
class Graph():
    def __init__(self, n, biDirectional):
    	lib.Graph_new.argtypes = [ctypes.c_int, ctypes.c_int]
    	lib.Graph_new.restypes = ctypes.c_void_p
    	# lib.Graph_new.restypes = POINTER(GRAPH_STRUCTURE)


    	lib.Graph_destroy.argtypes = [ctypes.c_void_p]
    	lib.Graph_destroy.restypes = ctypes.c_void_p

    	lib.Graph_print.argtypes = [ctypes.c_void_p]
    	lib.Graph_print.restypes = ctypes.c_void_p

    	lib.Graph_countEdge.argtypes = [ctypes.c_void_p]
    	lib.Graph_countEdge.restypes = ctypes.c_int

    	lib.Graph_setEdge.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    	lib.Graph_setEdge.restypes = ctypes.c_void_p

    	lib.Graph_check.argtypes = [ctypes.c_void_p]
    	lib.Graph_check.restypes = ctypes.c_void_p

    	self.obj = lib.Graph_new(n, biDirectional)

        # lib.Foo_new.argtypes = [ctypes.c_int]
        # lib.Foo_new.restype = ctypes.c_void_p

        # lib.Foo_bar.argtypes = [ctypes.c_void_p]
        # lib.Foo_bar.restype = ctypes.c_void_p

        # lib.Foo_foobar.argtypes = [ctypes.c_void_p, ctypes.c_int]
        # lib.Foo_foobar.restype = ctypes.c_int
		

    def destroy(self):
        lib.Graph_destroy(self.obj)

    def print(self):
    	lib.Graph_print(self.obj)

    def countEdge(self):
    	return lib.Graph_countEdge(self.obj)		
    
    def set_Edge(self, x, y):
    	lib.Graph_setEdge(self.obj, x, y)

    def check(self):
    	lib.Graph_check(self.obj)

    # def foobar(self, val):
    #     return lib.Foo_foobar(self.obj, val)


g1 = Graph(5, 1)
g1.check()
g1.print()
# print("Edge Count: ", g1.countEdge())
# g1.set_Edge(0, 1)
# g1.set_Edge(1, 2)
# g1.set_Edge(3, 4)
# print("Edge Count: ", g1.countEdge())
# g1.destroy()

