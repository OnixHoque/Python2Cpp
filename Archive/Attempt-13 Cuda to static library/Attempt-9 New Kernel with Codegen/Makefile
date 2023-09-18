TEMP_PATH = ./kernels/tmp

all:
	nvcc -Xcompiler -fPIC  -rdc=true -c -o $(TEMP_PATH)/temp.o fusedmm.cu
	nvcc -Xcompiler -fPIC -dlink -o $(TEMP_PATH)/temp2.o $(TEMP_PATH)/temp.o -lcudart
	ar cru ./bin/fusedmm_gpu.a $(TEMP_PATH)/*.o
	ranlib ./bin/fusedmm_gpu.a
	g++ timer.cpp -L./bin -l:fusedmm_gpu.a -o timer -L/usr/local/cuda/lib64 -lcudart

codegen:
	"Auto-tuning..."
	python3 ./kernels/codegen.py
	"Done!"
	
clean:
	rm -f *.a *.o main
	
