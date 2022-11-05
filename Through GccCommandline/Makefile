build:
	g++ -c -fPIC mygraph.cpp -o mygraph.o
	g++ -shared -Wl,-soname,libmygraph.so -o libmygraph.so mygraph.o

test: build
	python3 test.py
clean:
	rm -f *.o *.so
	rm -frd ./__pycache__
