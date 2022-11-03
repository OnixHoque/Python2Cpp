import ctypes
import numpy
import glob
from ctypes import *
# find the shared library, the path depends on the platform and Python version
libfile = glob.glob('./libmygraph.so')[0]

# 1. open the shared library
lib = ctypes.CDLL(libfile)


class Graph(object):
    def __init__(self, val):
        lib.Graph_new.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.Graph_new.restype = ctypes.c_void_p
        # lib.Graph_new.restypes = POINTER(GRAPH_STRUCTURE)


        lib.Graph_destroy.argtypes = [ctypes.c_void_p]
        lib.Graph_destroy.restype = ctypes.c_void_p

        lib.Graph_print.argtypes = [ctypes.c_void_p]
        lib.Graph_print.restype = ctypes.c_void_p

        lib.Graph_countEdge.argtypes = [ctypes.c_void_p]
        lib.Graph_countEdge.restype = ctypes.c_int

        lib.Graph_setEdge.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.Graph_setEdge.restype = ctypes.c_void_p

        lib.Graph_check.argtypes = [ctypes.c_void_p]
        lib.Graph_check.restype = ctypes.c_void_p

        self.obj = lib.Graph_new(val, 1)

    def printGraph(self):
        lib.Graph_print(self.obj)
    
    def setEdge(self, i, j):
        lib.Graph_setEdge(self.obj, i, j)

    def countEdge(self):
        return lib.Graph_countEdge(self.obj)

    def destroy(self):
        lib.Graph_destroy(self.obj)


g1 = Graph(5)
# g1.check()
g1.printGraph()
print("Edge Count: ", g1.countEdge())
g1.setEdge(0, 1)
g1.setEdge(1, 2)
g1.setEdge(3, 4)
print("Edge Count: ", g1.countEdge())
g1.destroy()

