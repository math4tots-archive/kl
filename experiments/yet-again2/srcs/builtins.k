#"no builtins"
from c import size_t
from c import printf
from c import malloc
from c import free
from c import strlen

extern char* KLC_CopyString(char const* s)

// Helper functions for implementing class List
extern void* KLC_realloc_var_array(void* buffer, size_t old_cap, size_t new_cap)
extern void KLC_partial_release_var_array(
    void* buffer, size_t size, size_t cap, void* delete_queue)
extern void KLC_var_array_clear_range(void* buffer, size_t begin, size_t end)
extern var KLC_var_array_get(void* buffer, size_t i)
extern void KLC_var_array_set(void* buffer, size_t i, var value)

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

  long normalMethod() {
    95189
  }

  size_t __GETsize() {
    _size
  }

  delete {
    free(static_cast(void*, this->_buffer))
  }
}


class List {
  size_t _size
  size_t _cap
  void* _buffer

  static List new() {
    $(List)
  }

  var __getitem(size_t i) {
    if (i >= _size) {
      throw "Index out of bounds"
    }
    KLC_var_array_get(_buffer, i)
  }

  var __setitem(size_t i, var value) {
    if (i >= _size) {
      throw "Index out of bounds"
    }
    KLC_var_array_set(_buffer, i, value)
    value
  }

  delete {
    KLC_partial_release_var_array(_buffer, _size, _cap, __delete_queue)
  }
}
