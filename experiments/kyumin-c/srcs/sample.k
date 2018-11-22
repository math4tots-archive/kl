#include <stdio.h>
#include "foo.k"

int printf(char* s, ...);

int main() {
  printf("Hello world!\n");
  printf("foo() = %d\n", foo());
  printf("foo() + 10 = %d\n", foo() + 10);
  printf("foo() - 10 = %d\n", foo() - 10);
  return 0;
}
