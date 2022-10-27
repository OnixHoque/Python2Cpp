from setuptools import setup, Extension

# Compile *cpp* into a shared library 
setup(
    #...
    ext_modules=[Extension('mygraphlib', ['mygraphlibrary.cpp'],),],
)
