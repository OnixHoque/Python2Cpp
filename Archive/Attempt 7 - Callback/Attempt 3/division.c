#include <stdio.h>

int x = 10;
int y = 20;

void perform_op(int (*op)(int, int ))
{
  int res = op(x, y);
  printf("Result is %d\n", res);
}
