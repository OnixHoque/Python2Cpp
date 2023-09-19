mkdir -p tmp bin
nvcc -Xcompiler -fPIC  -rdc=true -c -o ./tmp/temp.o adder.cu
nvcc -Xcompiler -fPIC -dlink -o ./tmp/temp2.o ./tmp/temp.o -lcudart
ar cru ./bin/cuda_static_lib.a ./tmp/*.o
ranlib ./bin/cuda_static_lib.a
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) wrapper.cpp -L./bin -l:cuda_static_lib.a -o myadder$(python3-config --extension-suffix) -L/usr/local/cuda/lib64 -lcudart