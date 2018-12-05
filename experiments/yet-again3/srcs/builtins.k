#"no builtins"
from c import size_t
from c import printf
from c import malloc
from c import free
from c import strlen
from c import cmalloc
from c import cfree
from c import crealloc
from c import strcpy
from c import strcat

extern c_char* KLC_CopyString(c_char const* s)

// For extracting type information
extern int KLC_get_tag(var x)
extern int KLC_get_obj_tag()
extern int KLC_get_bool_tag()
extern int KLC_get_int_tag()
extern int KLC_get_float_tag()
extern int KLC_get_type_tag()
extern c_char const* KLC_get_type_module_name(type t)
extern c_char const* KLC_get_type_short_name(type t)
extern type KLC_get_class(var x)

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
    (
      String.fromCString("<")
        .__add(str(KLC_get_class(this)))
        .__add(String.fromCString(" instance>"))
    )
  }

  String __str() {
    this.__repr()
  }
}

class StringBuilder(Object) {
  c_char* _buffer
  int _size
  int _cap

  static StringBuilder new() {
    $(StringBuilder)
  }

  void addString(String s) {
    _addCString(this, s->_buffer, s->_size)
  }

  void add(var x) {
    addString(str(x))
  }

  String __repr() {
    String.fromCString("<StringBuilder>")
  }

  String __str() {
    String.fromCString(_buffer)
  }

  static void _addCString(
      StringBuilder sb, c_char const* cstr, int added_size) {
    if (sb->_size + added_size + 1 >= sb->_cap) {
      sb->_cap = 2 * (sb->_size + added_size + 1)
      sb->_buffer = crealloc(sb->_buffer, sb->_cap)
    }
    strcpy(sb->_buffer + sb->_size, cstr)
    sb->_size = sb->_size + added_size
  }

  delete {
    free(static_cast(void*, this->_buffer))
  }
}

class String(Object) {
  c_char* _buffer
  int _size

  static String new(c_char const* cstr) {
    String.fromCString(cstr)
  }

  static String fromCString(c_char const* cstr) {
    String._fromBuffer(KLC_CopyString(cstr), strlen(cstr))
  }

  static String _fromBuffer(c_char* buffer, int size) {
    String ret = $(String)
    ret->_buffer = buffer
    ret->_size = size
    ret
  }

  String __add(String other) {
    int new_size = _size + other->_size
    c_char* new_buffer = cmalloc(new_size + 1)
    strcpy(new_buffer, _buffer)
    strcat(new_buffer, other->_buffer)
    String._fromBuffer(new_buffer, new_size)
  }

  c_long normalMethod() {
    95189
  }

  int __GETsize() {
    _size
  }

  String __str() {
    this
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

String get_type_str(type t) {
  (
    String.fromCString("<type ")
      .__add(String.fromCString(KLC_get_type_module_name(t)))
      .__add(String.fromCString("#"))
      .__add(String.fromCString(KLC_get_type_short_name(t)))
      .__add(String.fromCString(">"))
  )
}

String str(var x) {
  int tag = KLC_get_tag(x)
  if (tag == KLC_get_type_tag()) {
    get_type_str(x)
  } else {
    String s = x.__str()
    s
  }
}

void print(var x) {
  printf("%s\n", str(x)->_buffer)
}
