c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) mygraph_pybind.cpp -o mygraph$(python3-config --extension-suffix)
