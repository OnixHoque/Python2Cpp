#MKLROOT = /opt/intel/mkl
MKLROOT = /N/u2/t/taufique/intel/mkl
SPLIB_ROOT = ../../
NIST_DIR = $(SPLIB_ROOT)/include/NIST

CC = g++ -fopenmp -fpermissive -std=c++11 -O3 -DMKL_ILP64 -m64
#CC = g++ -fopenmp -fpermissive -std=c++11 -DMKL_ILP64 -DBREAKDOWN -m64
#CC = g++ -fopenmp -fpermissive -std=c++11 -DMKL_ILP64 -m64 -g -Og -mavx # Use this if binary is used with Valgrind
INC = -I${MKLROOT}/include
LIB = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
	  ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a \
	  ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp \
	  -lpthread -lm -ldl

mmio.o: $(NIST_DIR)/mmio.c
	$(CC) -I$(NIST_DIR) -c -o mmio.o $(NIST_DIR)/mmio.c

mmio.o: $(NIST_DIR)/mmio.c
	$(CC) -I$(NIST_DIR) -c -o mmio.o $(NIST_DIR)/mmio.c

hash-regular-static.o: hash-regular-static.cpp
	$(CC) -I${MKLROOT}/include -c $< -o $@

hash-regular-static: hash-regular-static.o mmio.o
	$(CC) $^ -o $@ $(LIB)

hash-regular-dynamic.o: hash-regular-dynamic.cpp
	$(CC) -I${MKLROOT}/include -c $< -o $@

hash-regular-dynamic: hash-regular-dynamic.o mmio.o
	$(CC) $^ -o $@ $(LIB)

hash-sliding-static.o: hash-sliding-static.cpp
	$(CC) -I${MKLROOT}/include -c $< -o $@

hash-sliding-static: hash-sliding-static.o mmio.o
	$(CC) $^ -o $@ $(LIB)

hash-sliding-dynamic.o: hash-sliding-dynamic.cpp
	$(CC) -I${MKLROOT}/include -c $< -o $@

hash-sliding-dynamic: hash-sliding-dynamic.o mmio.o
	$(CC) $^ -o $@ $(LIB)

heap-dynamic.o: heap-dynamic.cpp
	$(CC) -I${MKLROOT}/include -c $< -o $@

heap-dynamic: heap-dynamic.o mmio.o
	$(CC) $^ -o $@ $(LIB)

heap-static.o: heap-static.cpp
	$(CC) -I${MKLROOT}/include -c $< -o $@

heap-static: heap-static.o mmio.o
	$(CC) $^ -o $@ $(LIB)

spa-dynamic.o: spa-dynamic.cpp
	$(CC) -I${MKLROOT}/include -c $< -o $@

spa-dynamic: spa-dynamic.o mmio.o
	$(CC) $^ -o $@ $(LIB)

spa-static.o: spa-static.cpp
	$(CC) -I${MKLROOT}/include -c $< -o $@

spa-static: spa-static.o mmio.o
	$(CC) $^ -o $@ $(LIB)

pairwise-serial-dynamic.o: pairwise-serial-dynamic.cpp
	$(CC) -I${MKLROOT}/include -c $< -o $@

pairwise-serial-dynamic: pairwise-serial-dynamic.o mmio.o
	$(CC) $^ -o $@ $(LIB)

pairwise-serial-static.o: pairwise-serial-static.cpp
	$(CC) -I${MKLROOT}/include -c $< -o $@

pairwise-serial-static: pairwise-serial-static.o mmio.o
	$(CC) $^ -o $@ $(LIB)

pairwise-tree-dynamic.o: pairwise-tree-dynamic.cpp
	$(CC) -I${MKLROOT}/include -c $< -o $@

pairwise-tree-dynamic: pairwise-tree-dynamic.o mmio.o
	$(CC) $^ -o $@ $(LIB)

pairwise-tree-static.o: pairwise-tree-static.cpp
	$(CC) -I${MKLROOT}/include -c $< -o $@

pairwise-tree-static: pairwise-tree-static.o mmio.o
	$(CC) $^ -o $@ $(LIB)

mkl-serial.o: mkl-serial.cpp
	$(CC) -I${MKLROOT}/include -c $< -o $@

mkl-serial: mkl-serial.o mmio.o
	$(CC) $^ -o $@ $(LIB)

mkl-tree.o: mkl-tree.cpp
	$(CC) -I${MKLROOT}/include -c $< -o $@

mkl-tree: mkl-tree.o mmio.o
	$(CC) $^ -o $@ $(LIB)

window.o: window.cpp
	$(CC) -I${MKLROOT}/include -c -o window.o window: window.o mmio.o
	$(CC) -o window window.o mmio.o

all: window hash-regular-static hash-regular-dynamic hash-sliding-static hash-sliding-dynamic pairwise-serial-static pairwise-serial-dynamic pairwise-tree-static pairwise-tree-dynamic mkl-serial mkl-tree spa-static spa-dynamic heap-dynamic heap-static

clean:
	rm -f exp_scaling.o 
	rm -f exp_scaling
	rm -f hash-regular-dynamic.o 
	rm -f hash-regular-dynamic
	rm -f hash-regular-static.o 
	rm -f hash-regular-static
	rm -f hash-sliding-dynamic.o 
	rm -f hash-sliding-dynamic
	rm -f hash-sliding-static.o 
	rm -f hash-sliding-static
	rm -f pairwise-serial-static.o
	rm -f pairwise-serial-static
	rm -f pairwise-serial-dynamic.o
	rm -f pairwise-serial-dynamic
	rm -f pairwise-tree-static.o
	rm -f pairwise-tree-static
	rm -f pairwise-tree-dynamic.o
	rm -f pairwise-tree-dynamic
	rm -f spa-static.o
	rm -f spa-static
	rm -f spa-dynamic.o
	rm -f spa-dynamic
	rm -f heap-static.o
	rm -f heap-static
	rm -f heap-dynamic.o
	rm -f heap-dynamic
	rm -f mkl-serial.o
	rm -f mkl-serial
	rm -f mkl-tree.o
	rm -f mkl-tree
	rm -f mmio.o
	rm -f comparison.o
	rm -f comparison
	rm -f window.o
	rm -f window
