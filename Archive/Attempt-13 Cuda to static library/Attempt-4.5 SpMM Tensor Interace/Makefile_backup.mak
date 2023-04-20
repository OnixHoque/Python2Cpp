all:
	nvcc -Xcompiler -fPIC  -rdc=true -c -o ./tmp/temp.o kernel.cu
	nvcc -Xcompiler -fPIC -dlink -o ./tmp/temp2.o ./tmp/temp.o -lcudart
	ar cru ./bin/libsamplegpu.a ./tmp/*.o
	ranlib ./bin/libsamplegpu.a
	g++ main.cpp -L./bin -lsamplegpu -o main -L/usr/local/cuda/lib64 -lcudart

clean:
	rm -f *.a *.o main
	
