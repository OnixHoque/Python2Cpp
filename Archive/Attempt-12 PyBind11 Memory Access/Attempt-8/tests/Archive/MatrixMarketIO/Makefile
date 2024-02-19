SPLIB_HOME = ../../

NIST_HOME = $(SPLIB_HOME)/include/NIST

CC = g++ -fopenmp -fpermissive -std=c++11 -g -Og -mavx

mmio.o: $(NIST_HOME)/mmio.c
	$(CC) -I$(NIST_HOME) -c -o mmio.o $(NIST_HOME)/mmio.c

MMRead.o: MMRead.cpp
	$(CC) -c -o MMRead.o MMRead.cpp

MMRead: MMRead.o mmio.o
	$(CC) -o MMRead MMRead.o mmio.o

all: MMRead

clean:
	rm MMRead.o 
	rm MMRead
