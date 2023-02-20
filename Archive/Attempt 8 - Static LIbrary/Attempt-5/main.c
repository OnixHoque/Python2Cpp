#include <stdio.h>
#include <dlfcn.h>





int main()
{
	int (* ptr)(int , int );
	void * mylib = dlopen("add.so", RTLD_LOCAL | RTLD_LAZY);
	
	ptr = dlsym(mylib, "add");
	int a = 50;
	int b = 30;
	printf("Add result = %d\n", ptr(a, b));
	dlclose(mylib);
	return 0;
}
