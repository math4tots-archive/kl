#"no builtins"
from c import printf
from c import sprintf
from c import malloc
from c import free
from c import strlen
from c import cmalloc
from c import cfree
from c import crealloc
from c import strcpy
from c import strcat
from c import strcmp

extern c_char* KLC_CopyString(c_char const* s)

extern void KLC_release_vars_queued_for_exit()

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

// Helper functions for implementing class Lambda
extern void KLC_free_lambda_capture(KLC_Lambda_capture* capture)

trait Object() {
  void someInstanceMethodOnObject() = (
    printf(c"Inside someInstanceMethodOnObject\n")
  )

  String __repr() = (
      String.fromCString(c"<")
        .__add(str(KLC_get_class(this)))
        .__add(String.fromCString(c" instance>"))
  )

  String __str() = this.__repr()
}

trait Iterator(Object) {
  void forEachRemaining(var f) {
    while (this.__next(f)) {}
  }
}

trait Iterable(Object) {

  Stream stream() = Stream.fromIterable(this)

  var to(var collector) = collector.fromIterable(this)

  var map(var f) {
    var that = this
    typeof(this).fromIterable(
      Stream(() => MappingIterator(that.__iter(), f)))
  }

  var filter(var f) {
    var that = this
    typeof(this).fromIterable(
      Stream(() => FilterIterator(that.__iter(), f)))
  }

  void forEach(var f) {
    this.__iter().forEachRemaining(f)
  }
}

class Stream(Iterable) {
  // A lazy Iterable defined by a function that knows
  // how to produce new iterators.

  var newIterator

  class Stream fromIterable(var iterable) {
    if (typeof(iterable) is Stream) {
      iterable
    } else {
      Stream(() => iterable.__iter())
    }
  }

  static Stream new(var newIterator) {
    Stream ret = $(Stream)
    ret->newIterator = newIterator
    ret
  }

  Stream stream() = this

  var __iter() = newIterator()

  Stream take(int n) {
    var iteratorFactory = newIterator
    Stream(() => TakeIterator(iteratorFactory(), n))
  }
}

class MappingIterator(Iterator) {
  var iterator
  var f

  static MappingIterator new(var iterator, var f) {
    MappingIterator ret = $(MappingIterator)
    ret->iterator = iterator
    ret->f = f
    ret
  }

  bool __next(var callback) {
    MappingIterator that = this
    iterator.__next(x => callback(that->f(x)))
  }
}

class FilterIterator(Iterator) {
  var iterator
  var f
  bool hasNext
  var next

  static FilterIterator new(var iterator, var f) {
    FilterIterator ret = $(FilterIterator)
    ret->iterator = iterator
    ret->f = f
    ret
  }

  bool __next(var callback) {
    List found = [false]
    FilterIterator that = this
    var process = def(var x) {
      if (that->f(x)) {
        found[0] = true
        callback(x)
      }
    }
    while (!found[0] && iterator.__next(process)) {
    }
    found[0]
  }
}

Stream range(int start, int end) {
  Stream(() => SimpleIntRangeIterator(start, end))
}

class SimpleIntRangeIterator(Iterator) {
  int i
  int limit

  static SimpleIntRangeIterator new(int start, int end) {
    SimpleIntRangeIterator iter = $(SimpleIntRangeIterator)
    iter->i = start
    iter->limit = end
    iter
  }

  bool __next(var callback) {
    if (i < limit) {
      callback(i)
      i = i + 1
      true
    } else {
      false
    }
  }
}

class TakeIterator(Iterator) {
  var iterator
  int remaining

  static TakeIterator new(var iterator, int remaining) {
    TakeIterator ret = $(TakeIterator)
    ret->iterator = iterator
    ret->remaining = remaining
    ret
  }

  bool __next(var callback) {
    if (remaining > 0) {
      if (iterator.__next(callback)) {
        remaining = remaining - 1
        true
      } else {
        remaining = 0
        false
      }
    } else {
      false
    }
  }
}

extern type typeof(var x) = KLC_get_class(x)

class assert() {
  static void new(bool cond) {
    if (!cond) {
      throw "assertion failed"
    }
  }

  static void equal(var left, var right) {
    if (left != right) {
      throw 'Expected ' + str(left) + ' to equal ' + str(right)
    }
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
    _addCStringWithSize(this, s->_buffer, s->_size)
  }

  void add(var x) {
    addString(str(x))
  }

  String __repr() {
    String.fromCString(c"<StringBuilder>")
  }

  String __str() {
    String.fromCString(_buffer)
  }

  static void _addCString(
      StringBuilder sb, c_char const* cstr) {
    _addCStringWithSize(sb, cstr, strlen(cstr))
  }

  static void _addCStringWithSize(
      StringBuilder sb, c_char const* cstr, int added_size) {
    if (sb->_size + added_size + 1 >= sb->_cap) {
      sb->_cap = 2 * (sb->_size + added_size + 1)
      sb->_buffer = crealloc(sb->_buffer, sb->_cap)
    }
    strcpy(sb->_buffer + sb->_size, cstr)
    sb->_size = sb->_size + added_size
  }

  delete {
    free(dynamic_cast(void*, this->_buffer))
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

  static c_char const* toCString(String s) = s->_buffer

  String __add(String other) {
    int new_size = _size + other->_size
    c_char* new_buffer = cmalloc(new_size + 1)
    strcpy(new_buffer, _buffer)
    strcat(new_buffer, other->_buffer)
    String._fromBuffer(new_buffer, new_size)
  }

  int normalMethod() {
    95189
  }

  int __GETsize() {
    _size
  }

  String __str() {
    this
  }

  bool __eq(String other) {
    _size == other->_size && strcmp(_buffer, other->_buffer) is 0
  }

  delete {
    free(dynamic_cast(void*, this->_buffer))
  }
}

class Lambda(Object) {
  KLC_Lambda_capture* _capture
  KLC_Lambda_body* _body

  extern __call = KLC_lambda_call

  delete {
    KLC_free_lambda_capture(_capture)
  }
}

Lambda `%lambda`(KLC_Lambda_capture* capture, KLC_Lambda_body* body) {
  Lambda ret = $(Lambda)
  ret->_capture = capture
  ret->_body = body
  ret
}

// This is an implicit function that will get called for
// normal string literals
String `%str`(c_char const* cstr) = String.fromCString(cstr)

String `%exception`(var s) = str(s)

bool `%not`(var x) {
  if (x) {
    false
  } else {
    true
  }
}

extern String KLC_str(c_char const* cstr) = String.fromCString(cstr)
extern c_char const* KLC_cstr(String s) = s->_buffer

class ListIterator(Iterator) {
  List list
  int i

  static ListIterator new(List list) {
    ListIterator ret = $(ListIterator)
    ret->list = list
    ret->i = 0
    ret
  }

  bool __next(var callback) {
    if (i < list->_size) {
      callback(list[i])
      i = i + 1
      true
    } else {
      false
    }
  }
}

class List(Iterable) {
  int _size
  int _cap
  void* _buffer

  class List unit(var x) = [x]

  class List fromIterable(var iterable) {
    List ret = []
    var iterator = iterable.__iter()
    iterator.forEachRemaining(x => ret.push(x))
    ret
  }

  static List new() = $(List)

  int __GETsize() = _size

  void push(var x) {
    if (_size >= _cap) {
      int old_cap = _cap
      _cap = _cap * 2 + 16
      _buffer = KLC_realloc_var_array(
        _buffer,
        dynamic_cast(size_t, old_cap),
        dynamic_cast(size_t, _cap),
      )
    }
    KLC_var_array_set(_buffer, dynamic_cast(size_t, _size), x)
    _size = _size + 1
  }

  bool __eq(List other) {
    _size == other->_size && ${
      int i = 0
      bool result = true
      while (i < _size) {
        if (this.__getitem(i) == other.__getitem(i)) {
          i = i + 1
        } else {
          result = false
          i = i + 1
        }
      }
      result
    }
  }

  var __getitem(int i) {
    if (i < 0) {
      i = i + _size
    }
    if (i < 0 || i >= _size) {
      throw "Index out of bounds"
    }
    KLC_var_array_get(_buffer, dynamic_cast(size_t, i))
  }

  var __setitem(int i, var value) {
    if (i < 0) {
      i = i + _size
    }
    if (i < 0 || i >= _size) {
      throw "Index out of bounds"
    }
    KLC_var_array_set(_buffer, dynamic_cast(size_t, i), value)
    value
  }

  String __repr() {
    StringBuilder sb = StringBuilder()
    int i = 0
    sb.add('[')
    while (i < _size) {
      if (i > 0) {
        sb.add(', ')
      }
      sb.add(this.__getitem(i))
      i = i + 1
    }
    sb.add(']')
    str(sb)
  }

  ListIterator __iter() = ListIterator(this)

  delete {
    KLC_partial_release_var_array(
      _buffer,
      dynamic_cast(size_t, _size),
      dynamic_cast(size_t, _cap),
      __delete_queue,
    )
  }
}

List `%mklist`() {
  List()
}

List `%listpush`(List list, var x) {
  list.push(x)
  list
}

bool `%eq`(var a, var b) {
  int tag_a = KLC_get_tag(a)
  int tag_b = KLC_get_tag(b)
  int obj_tag = KLC_get_obj_tag()
  int int_tag = KLC_get_int_tag()
  int float_tag = KLC_get_float_tag()
  if (a is b) {
    dynamic_cast(bool, 1)
  } else if (tag_a is int_tag && tag_b is float_tag ||
             tag_a is float_tag && tag_b is int_tag) {
    float fa = a
    float fb = b
    fa is fb
  } else if (tag_a != tag_b) {
    dynamic_cast(bool, 0)
  } else if (tag_a == obj_tag && tag_b == obj_tag) {
    type ca = KLC_get_class(a)
    type cb = KLC_get_class(b)
    if (ca is cb) {
      dynamic_cast(bool, a.__eq(b))
    } else {
      dynamic_cast(bool, 0)
    }
  } else {
    dynamic_cast(bool, 0)
  }
}

bool `%ne`(var a, var b) {
  !(a == b)
}

String get_type_str(type t) = (
    String.fromCString(c"<type ")
      .__add(String.fromCString(KLC_get_type_module_name(t)))
      .__add(String.fromCString(c"#"))
      .__add(String.fromCString(KLC_get_type_short_name(t)))
      .__add(String.fromCString(c">"))
)

String get_int_str(int x) {
  // TODO: Make implementation more portable
  c_char* buffer = cmalloc(64)
  sprintf(buffer, c"%lu", x)
  String ret = String.fromCString(buffer)
  cfree(buffer)
  ret
}

String get_float_str(float x) {
  // TODO: Make implementation more portable
  c_char* buffer = cmalloc(64)
  sprintf(buffer, c"%f", x)
  String ret = String.fromCString(buffer)
  cfree(buffer)
  ret
}

String get_bool_str(bool x) {
  if (x) {
    "true"
  } else {
    "false"
  }
}

String str(var x) {
  int tag = KLC_get_tag(x)
  if (tag == KLC_get_type_tag()) {
    get_type_str(x)
  } else if (tag == KLC_get_int_tag()) {
    get_int_str(x)
  } else if (tag == KLC_get_float_tag()) {
    get_float_str(x)
  } else if (tag == KLC_get_bool_tag()) {
    get_bool_str(x)
  } else {
    String s = x.__str()
    s
  }
}

void print(var x) {
  printf(c"%s\n", str(x)->_buffer)
}
