# import sys
# from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension("mycoomatrix",
        ["csrc/mycoomatrix.cpp"],
	    extra_objects=[],
        extra_compile_args=[], 
        extra_link_args=[],
        ),
]

setup(
    name="mycoomatrix",
    ext_modules=ext_modules,
    zip_safe=False,
setup_requires=['pybind11>=2.2'],
    python_requires=">=3.7"
)
