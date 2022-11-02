g++ -c -fPIC mygraph.cpp -o mygraph.o
g++ -shared -Wl,-soname,libmygraph.so -o libmygraph.so mygraph.o

