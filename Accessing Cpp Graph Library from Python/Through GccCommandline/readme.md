## Compile the shared object using **gcc**.

1. Run the following commands to generate shared object from mygraph.cpp file.

- `g++ -c -fPIC mygraph.cpp -o mygraph.o`
- `g++ -shared -Wl,-soname,libmygraph.so -o libmygraph.so mygraph.o`

|| Note: The generated shared object file's name should start with *lib*

2. Run `test.py` to see check if the graph is being used correctly. The mygraph.py works as a wrapper for the cpp library.


--

Special thanks to the following tutorial: https://www.auctoris.co.uk/2017/04/29/calling-c-classes-from-python-with-ctypes/

Read More: https://tldp.org/HOWTO/Program-Library-HOWTO/shared-libraries.html
