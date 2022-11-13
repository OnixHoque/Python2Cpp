all:
	g++ -c -fPIC mymath.c
	ar rvs mymath.a mymath.o
	g++ main.c mymath.o

clean_partial:
	rm *.out *.o

clean:
	rm *.out *.a *.o

