#include <stdio.h>
// #include "add.c"
// #include "sub.c"

int add(int, int);
int sub(int, int);

int main()
{
	int x = add(10, 20);
	int y = sub(50, 20);
	printf("add 10+20, result=%d\n", x);
	printf("sub 50-20, result=%d\n", y);
	return 0;
}
