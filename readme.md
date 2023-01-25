# Accessing C++ Library from Python Code

[New! PyBind11 example added!]

This repo demonstrats how we can use a C++ library from Python by generating a Cpp shared object. This can be done directly by adding a few extra flags in gcc, or using setuptools library provided in Python. The later is a much cleaner approach. We can also use PyBind11, which simplifies the process greatly!

**New: Added support for passing function pointer! See performOp() function in `mygraph.py`**

The Cpp library is a simple Graph class that creates adjacency matrix dynamically through constructor. It also has other utility functions for printing the graph, setting edge, and freeing/disposing the dynamic memory.

## Compile the shared object using gcc.

1. Run the following commands to generate shared object from mygraph.cpp file (or just run the `make` command).

- `g++ -c -fPIC mygraph.cpp -o mygraph.o`
- `g++ -shared -Wl,-soname,libmygraph.so -o libmygraph.so mygraph.o`

	Note: Conventionally, the generated shared object file's name starts with *lib*.

2. Run `test.py` to check if the graph is being used correctly (or just run the `make test` command). The `mygraph.py` works as a wrapper for the cpp library.

## Compile the shared object through Python setuptools

1. Run the following command to generate shared object from mygraph.cpp file (or just run the `make` command). It will be generated in `\build\*\` folder. In this case, the name of the shared object is automatically generated. 

- `python3 setup.py build`
or
- `python setup.py build`
or
- `py setup.py build`

2. Run `test.py` to check if the graph is being used correctly (or just run the `make test` command). The `mygraph.py` works as a wrapper for the cpp library.

## ***[New!]*** Compile the shared object through PyBind11 - to support C++11 features 

1. Make sure that you have PyBind11 installed (`pip install pybind11`).
2. Run `./make.sh`
3. Run `python3 test.py` or `python test.py`

Read the PyBind11 Documentation for more functionalities: 
- https://pybind11.readthedocs.io/en/stable/classes.html
- https://pybind11.readthedocs.io/en/stable/advanced/functions.html
- https://pybind11.readthedocs.io/en/stable/advanced/classes.html

--

## Special thanks to the following tutorials
- https://nesi.github.io/perf-training/python-scatter/ctypes
- https://www.auctoris.co.uk/2017/04/29/calling-c-classes-from-python-with-ctypes/
- https://gist.github.com/Nican/5198719
- https://princekfrancis.medium.com/passing-a-callback-function-from-python-to-c-351ac944e041

Read More: https://tldp.org/HOWTO/Program-Library-HOWTO/shared-libraries.html
