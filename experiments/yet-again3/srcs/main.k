#include <stdio.h>
from lib import sayhi
from lib import printf
from lib import Baz
from c import size_t

extern void* malloc(size_t size)
extern void free(void* p)

struct Foo {
  Baz b
  c_int x
  extern c_long y
}

extern void asdf(c_unsigned_char c) {
}

struct C {
  c_int d
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

extern c_int main() {
  Foo foo
  printf("foo.x = %d\n", foo.x)
  foo.x = 15
  printf("foo.x = %d\n", foo.x)
  void* fp = malloc(static_cast(size_t, 3))
  c_unsigned_long i = static_cast(c_unsigned_long, 14)
  printf("fp = %p\n", fp)
  free(fp)

  ${
    c_signed_char c
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
  static_cast(c_int, 0)
}
