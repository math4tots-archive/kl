#"no builtins"
from c import size_t
from c import printf
from c import malloc
from c import free
from c import strlen
from c import cmalloc
from c import cfree

extern c_char* KLC_CopyString(c_char const* s)

// Helper functions for implementing class List
extern void* KLC_realloc_var_array(void* buffer, size_t old_cap, size_t new_cap)
extern void KLC_partial_release_var_array(
    void* buffer, size_t size, size_t cap, void* delete_queue)
extern void KLC_var_array_clear_range(void* buffer, size_t begin, size_t end)
extern var KLC_var_array_get(void* buffer, size_t i)
extern void KLC_var_array_set(void* buffer, size_t i, var value)

trait Object() {
  int someInstanceMethodOnObject() {
    printf("Inside someInstanceMethodOnObject\n")
  }

  String __repr() {
    this.__str()
  }
}

class String(Object) {
  c_char* _buffer
  int _size

  static String new(c_char const* cstr) {
    String.fromCString(cstr)
  }

  static String fromCString(c_char const* cstr) {
    String ret = $(String)
    ret->_buffer = KLC_CopyString(cstr)
    ret->_size = static_cast(int, strlen(cstr))
    ret
  }

  c_long normalMethod() {
    95189
  }

  int __GETsize() {
    _size
  }

  delete {
    free(static_cast(void*, this->_buffer))
  }
}

class List(Object) {
  int _size
  int _cap
  void* _buffer

  static List new() {
    $(List)
  }

  int __GETsize() {
    _size
  }

  void push(var x) {
    if (_size >= _cap) {
      int old_cap = _cap
      _cap = _cap * 2 + 16
      _buffer = KLC_realloc_var_array(
        _buffer,
        static_cast(size_t, old_cap),
        static_cast(size_t, _cap),
      )
    }
    KLC_var_array_set(_buffer, static_cast(size_t, _size), x)
    _size = _size + 1
  }

  var __getitem(int i) {
    if (i < 0 || i >= _size) {
      throw "Index out of bounds"
    }
    KLC_var_array_get(_buffer, static_cast(size_t, i))
  }

  var __setitem(int i, var value) {
    if (i < 0 || i >= _size) {
      throw "Index out of bounds"
    }
    KLC_var_array_set(_buffer, static_cast(size_t, i), value)
    value
  }

  delete {
    KLC_partial_release_var_array(
      _buffer,
      static_cast(size_t, _size),
      static_cast(size_t, _cap),
      __delete_queue,
    )
  }
}
