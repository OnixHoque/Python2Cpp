#include <stdint.h>
#include <stdio.h>

struct mes_t
{
    uint32_t field1;
    uint32_t field2;
    void* data;
};

typedef int function_callback(struct mes_t* message );

function_callback* my_callback;

int function_one(function_callback fcb){
	//Set to a global variable for later use
	my_callback = fcb;

	//Declare object in stack
	struct mes_t mes;
	mes.field1 = 132;
	mes.field2 = 264;
	mes.data = NULL;

	//Pass pointer of object in stack, and print the return value
	printf("Got from python: %d\n", my_callback( &mes ) );
}
