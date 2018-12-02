#"no builtins"
from c import size_t
from c import printf
from c import malloc
from c import free
from c import strlen

extern char* KLC_CopyString(char const* s)

class String {
  char* _buffer
  size_t _size

  static String new(char const* cstr) {
    String.fromCString(cstr)
  }

  static String fromCString(char const* cstr) {
    String ret = $(String)
    ret->_buffer = KLC_CopyString(cstr)
    ret->_size = strlen(cstr)
    ret
  }

  size_t getSize() {
    _size
  }

  delete {
    free(static_cast(void*, this->_buffer))
  }
}
