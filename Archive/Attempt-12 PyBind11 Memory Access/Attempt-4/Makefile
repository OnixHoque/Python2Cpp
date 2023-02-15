all:
	c++ -O3 -Wall -shared -std=c++11 -fPIC $(shell python3 -m pybind11 --includes) mycoomatrix.cpp -o mycoomatrix$(shell python3-config --extension-suffix)
test:
	python3 test.py
clean:
	rm -f *.so

