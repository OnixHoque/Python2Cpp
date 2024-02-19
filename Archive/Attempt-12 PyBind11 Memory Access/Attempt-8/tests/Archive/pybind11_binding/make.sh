c++ -O3 -Wall -shared -std=c++11 -fopenmp -fPIC $(python3 -m pybind11 --includes) COO_pybind.cpp -o COO$(python3-config --extension-suffix)
c++ -O3 -Wall -shared -std=c++11 -fopenmp -fPIC $(python3 -m pybind11 --includes) CSC_pybind.cpp -o CSC$(python3-config --extension-suffix)
