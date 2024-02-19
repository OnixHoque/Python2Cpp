# import sys
# from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension("pysplib.csplib",
        ["csrc/pybind_wrapper.cpp"],
	    extra_objects=[],
	extra_compile_args=["-O3", "-march=native", "-Wall", "-std=c++11", "-fopenmp"],
        extra_link_args=['-lgomp'],
        ),
]

setup(
    name="pysplib",
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.7"
)
