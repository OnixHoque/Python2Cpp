import ctypes
import glob

libfile = glob.glob('build/*/mygraph*.so')[0]
lib = ctypes.CDLL(libfile)

class Graph(object):
    def __init__(self, val):
        lib.Graph_new.argtypes = [ctypes.c_int]
        lib.Graph_new.restype = ctypes.c_void_p

        lib.Graph_print.argtypes = [ctypes.c_void_p]
        lib.Graph_print.restype = ctypes.c_void_p
        
        lib.Graph_setEdge.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.Graph_setEdge.restype = ctypes.c_void_p

        lib.Graph_countEdge.argtypes = [ctypes.c_void_p]
        lib.Graph_countEdge.restype = ctypes.c_int

        lib.Graph_performOp.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.Graph_performOp.restype = ctypes.c_void_p

        lib.Graph_destroy.argtypes = [ctypes.c_void_p]
        lib.Graph_destroy.restype = ctypes.c_void_p
        
        self.obj = lib.Graph_new(val)
        self.FUNCTION_CAST = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)

    def printGraph(self):
        lib.Graph_print(self.obj)
    
    def setEdge(self, i, j):
        lib.Graph_setEdge(self.obj, i, j)

    def countEdge(self):
        return lib.Graph_countEdge(self.obj)

    def performOp(self, op):
        lib.Graph_performOp(self.obj, self.FUNCTION_CAST(op))

    def destroy(self):
        lib.Graph_destroy(self.obj)
