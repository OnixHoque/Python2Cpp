gcc  -c -fPIC add.c -o ./lib/add.o
gcc  -c -fPIC sub.c -o ./lib/sub.o
ar rcs build/math.a ./lib/*

c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) mymath.cpp build/math.a -o mymath$(python3-config --extension-suffix)
