#include <stdio.h>
from lib import sayhi
from lib import printf
from lib import Baz
from c import size_t

extern void* malloc(size_t size)

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
  void* fp = malloc(3)
  unsigned long i = 14
  printf("fp = %p\n", fp)
  sayhi()
  0
}
