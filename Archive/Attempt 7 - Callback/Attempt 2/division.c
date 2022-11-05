#include <stdio.h>
void divide(void (*ptr)(int *, int *), int a, int b){
  int s = a / b;
  int r = a % b;
  (*ptr) (&s, &r);
}
void print_sum(int * s, int * r){
 printf("Quotient is %d, remainder is %d\n", *s, *r);
}
int main(){
  void (*ptr)() = &print_sum;
  divide(ptr, 7, 4);
}
