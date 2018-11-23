#include <stdio.h>
from foo import foo
from foo import bar

extern int printf(char* s, ...);

extern int main() {
  printf("Hello world!\n");
  printf("foo() = %d\n", foo());
  printf("foo() + 10 = %d\n", foo() + 10);
  printf("foo() - 10 = %d\n", foo() - 10);
  printf("bar() - 10 = %d\n", bar() - 10);
  return 0;
}
