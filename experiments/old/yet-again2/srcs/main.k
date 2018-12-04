#include <stdio.h>
from lib import sayhi
from lib import printf
from lib import Baz
from c import size_t

extern void* malloc(size_t size)
extern void free(void* p)

struct Foo {
  Baz b
  int x
  extern long y
}

extern void asdf(unsigned char c) {
}

struct C {
  int d
}

struct B {
  C c
}

struct A {
  B b
}

class Animal {
  A a
  Food f
}

class Food {
}

extern int main() {
  Foo foo
  printf("foo.x = %d\n", foo.x)
  foo.x = 15
  printf("foo.x = %d\n", foo.x)
  void* fp = malloc(3)
  unsigned long i = 14
  printf("fp = %p\n", fp)
  free(fp)

  ${
    signed char c
    asdf(c)
  }

  ${
    A a
    printf("a.b.c.d = %d\n", a.b.c.d)
    a.b.c.d = 149
    printf("a.b.c.d = %d\n", a.b.c.d)
  }

  ${
    Animal a = $(Animal)
    Animal b = a
    // printf("a.f = %p\n", a.f)
  }

  sayhi()
  0
}
