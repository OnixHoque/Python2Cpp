# Accessing Cpp Graph Library from Python

This repo demonstrats how we can use a Cpp library from Python by generating a shared object of the Cpp file. This can be done directly by adding a few extra flags in gcc, or using setuptools library provided in Python. The later is a more cleaner approach.

The Cpp library is a simple Graph class that creates adjacency matrix dynamically through constructor. It also has other utility functions for printing the graph, setting edge, and freeing/disposing the dynamic memory.

## Compile the shared object using gcc.

1. Run the following commands to generate shared object from mygraph.cpp file.

- `g++ -c -fPIC mygraph.cpp -o mygraph.o`
- `g++ -shared -Wl,-soname,libmygraph.so -o libmygraph.so mygraph.o`

	Note: The generated shared object file's name should start with *lib*

2. Run `test.py` to check if the graph is being used correctly. The `mygraph.py` works as a wrapper for the cpp library.

## Compile the shared object through Python setuptools

1. Run the following command to generate shared object from mygraph.cpp file. It will be generated in `\build\*\` folder. In this case, the name of the shared object is automatically generated.

- `python3 setup.py build`
or
- `python setup.py build`
or
- `py setup.py build`

2. Run `test.py` to check if the graph is being used correctly. The `mygraph.py` works as a wrapper for the cpp library.



## Special thanks to the following tutorials:
- https://nesi.github.io/perf-training/python-scatter/ctypes
- https://www.auctoris.co.uk/2017/04/29/calling-c-classes-from-python-with-ctypes/

Read More: https://tldp.org/HOWTO/Program-Library-HOWTO/shared-libraries.html
