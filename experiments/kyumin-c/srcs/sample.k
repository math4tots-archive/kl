#include <stdio.h>
from foo import foo
from foo import bar
from foo import Foo

extern int printf(char* s, ...)

int f(Foo foo) {
  return 123
}

extern int main() {
  printf(@"Hello world!\n")
  printf(@"foo() = %d\n", foo())
  printf(@"foo() + 10 = %d\n", foo() + 10)
  printf(@"foo() - 10 = %d\n", foo() - 10)
  printf(@"bar() - 10 = %d\n", bar() - 10)
  return 0
}
