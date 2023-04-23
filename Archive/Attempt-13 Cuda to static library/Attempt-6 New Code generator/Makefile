TEMP_PATH = ./kernels/tmp

all:
	echo "Auto-tuning..."
	python3 ./kernels/codegen.py
	echo "Done!"
	nvcc -Xcompiler -fPIC  -rdc=true -c -o $(TEMP_PATH)/temp.o fusedmm.cu
	nvcc -Xcompiler -fPIC -dlink -o $(TEMP_PATH)/temp2.o $(TEMP_PATH)/temp.o -lcudart
	ar cru ./bin/fusedmm_gpu.a $(TEMP_PATH)/*.o
	ranlib ./bin/fusedmm_gpu.a
	g++ main.cpp -L./bin -l:fusedmm_gpu.a -o main -L/usr/local/cuda/lib64 -lcudart

clean:
	rm -f *.a *.o main
	
