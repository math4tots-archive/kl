#include <stdio.h>
from lib import sayhi
from lib import printf
from lib import Baz

struct Foo {
  Baz b
  int x
  extern long y
}

extern int main() {
  Foo foo
  printf("foo.x = %d\n", foo.x)
  foo.x = 15
  printf("foo.x = %d\n", foo.x)
  sayhi()
  0
}
