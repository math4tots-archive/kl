from c import size_t
from c import printf
from c import malloc
from c import free

extern char* KLC_CopyString(char const* s)

class String {
  char* buffer
  size_t size

  delete {
    printf("String:delete called\n")
    printf("buffer = %p\n", static_cast(void*, this.buffer))
    printf("buffer = %s\n", this.buffer)
    free(static_cast(void*, this.buffer))
  }
}

extern int main(int argc, char** argv) {
  String s = $(String)
  s.buffer = KLC_CopyString("Hello world!")
  0
}
