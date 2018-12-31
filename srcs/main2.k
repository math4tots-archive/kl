from c import size_t
from c import printf
from c import malloc
from c import free

extern c_char* KLC_CopyString(c_char const* s)

class Foo {}

extern c_int main(c_int argc, c_char** argv) {
  String s = String("Hello world!")
  String s2 = String.fromCString("Hello world2")
  var x = 10
  var y = s
  c_int xi = x
  var vf = 1.4
  c_double f = vf
  String s3 = y
  printf("2 + 3 = %d\n", dynamic_cast(c_int, 2 + 3))
  printf("2 + -3 = %d\n", dynamic_cast(c_int, 2 + -3))
  printf("2 < 3 = %d\n", dynamic_cast(c_int, 2 < 3))
  printf("2 > 3 = %d\n", dynamic_cast(c_int, 2 > 3))
  printf("0 || 2 = %d\n", 0 || 2)
  printf("0 && 2 = %d\n", 0 && 2)
  bool b = 10
  if (5 < 10) {
    printf("5 < 10 is true\n")
  } else {
    printf("5 < 10 is false\n")
  }
  recursive_print(0, 10)
  c_char const* ss = if (1) {
    "1 is true"
  } else {
    "1 is false"
  }
  printf("ss = %s\n", ss)
  printf("Before\n")
  Foo foo = $(Foo)
  var vfoo = foo
  ${
    c_long normalMethod = s.normalMethod()
    printf("s.normalMethod() = %lu\n", normalMethod)
    c_long size = s.size
    printf("size = %ld\n", size)
  }
  ${
    Foo foo2 = vfoo
    Foo f3 = y
  }
  printf("After\n")
  dynamic_cast(c_int, 0)
}

void recursive_print(int i, int limit) {
  if (i < limit) {
    printf("i = %d\n", dynamic_cast(c_int, i))
    recursive_print(i + 1, limit)
  }
}
