#"no builtins"
from c import size_t
from c import printf
from c import malloc
from c import free
from c import strlen

extern char* KLC_CopyString(char const* s)

class String {
  char* buffer
  size_t size

  static String new(char const* cstr) {
    printf("new called!\n")
    String.fromCString(cstr)
  }

  static String fromCString(char const* cstr) {
    printf("fromCString called!\n")
    String ret = $(String)
    ret.buffer = KLC_CopyString(cstr)
    ret.size = strlen(cstr)
    ret
  }

  delete {
    printf("String:delete called\n")
    printf("buffer = %p\n", static_cast(void*, this.buffer))
    printf("buffer = %s\n", this.buffer)
    free(static_cast(void*, this.buffer))
  }
}
