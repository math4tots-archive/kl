from c import size_t
from c import printf
from c import malloc
from c import free

extern char* KLC_CopyString(char const* s)

extern int main(int argc, char** argv) {
  String s = String("Hello world!")
  String s2 = String.fromCString("Hello world2")
  0
}
