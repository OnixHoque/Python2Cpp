gcc -c -fPIC mymath/add.c -o ./mymath/add.o
gcc -c -fPIC mymath/sub.c -o ./mymath/sub.o
ar rcs ./mymath/mymath.a ./mymath/*.o
echo "Generated Static library: mymath/mymath.a"
pip install .
cp build/lib*/*.so .
# c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) mymath.cpp build/math.a -o mymath$(python3-config --extension-suffix)
