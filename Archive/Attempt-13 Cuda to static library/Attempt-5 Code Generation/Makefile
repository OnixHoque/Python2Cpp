all:
	nvcc -Xcompiler -fPIC -rdc=true -c -o ./tmp/*.o ./src/*.cu
	nvcc -Xcompiler -fPIC -dlink -o ./tmp/2/*.o ./tmp/*.o -lcudart
	ar cru ./bin/libsamplegpu.a ./tmp/2/*.o ./tmp/*.o
	ranlib ./bin/libsamplegpu.a
	g++ ./tests/main.cpp -L./bin -lsamplegpu -o ./tests/main -L/usr/local/cuda/lib64 -lcudart

clean:
	rm -f *.a *.o main
	
