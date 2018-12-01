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
    String.fromCString(cstr)
  }

  static String fromCString(char const* cstr) {
    String ret = $(String)
    ret.buffer = KLC_CopyString(cstr)
    ret.size = strlen(cstr)
    ret
  }

  delete {
    free(static_cast(void*, this.buffer))
  }
}
