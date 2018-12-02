from c import size_t
from c import printf
from c import malloc
from c import free

extern char* KLC_CopyString(char const* s)

class Foo {}

extern int main(int argc, char** argv) {
  String s = String("Hello world!")
  String s2 = String.fromCString("Hello world2")
  var x = 10
  var y = s
  int xi = x
  var vf = 1.4
  double f = vf
  String s3 = y
  printf("Before\n")
  Foo foo = $(Foo)
  var vfoo = foo
  ${
    long normalMethod = s.normalMethod()
    printf("s.normalMethod() = %lu\n", normalMethod)
    long size = s.size
    printf("size = %ld\n", size)
  }
  ${
    Foo foo2 = vfoo
    Foo f3 = y
  }
  printf("After\n")
  0
}
