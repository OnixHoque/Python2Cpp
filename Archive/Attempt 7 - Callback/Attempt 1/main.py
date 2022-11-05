import ctypes

testlib = ctypes.CDLL('./testlib.so')

#Declare the data structure
class mes_t(ctypes.Structure):
    _fields_ = (
        ('field1', ctypes.c_uint32),
        ('field2', ctypes.c_uint32),
        ('data', ctypes.c_void_p))

#Declare the callback type, since that is not stored in the library
callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(mes_t) )

def the_callback(mes_p):
    #dereference the pointer
    my_mes = mes_p[0]

    print( "I got a mes_t object! mes.field1=%r, mes.field2=%r, mes.data=%r" \
          % (my_mes.field1, my_mes.field2, my_mes.data))

    #Return some random value
    return 999

#Let the library know about the callback function by calling "function_one"
result = testlib.function_one(callback_type(the_callback) )
