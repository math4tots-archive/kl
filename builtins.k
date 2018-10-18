
#(define C_HEADERS [])
#(define C_SOURCES [
  'main.c'    // This is where the main program is
])

// Frameworks that are required for deploying to darwin targets.
#(define APPLE_FRAMEWORKS [])

// Special functions required by the language
bool _Is(var a, var b)
bool _IsNot(var a, var b)
bool _Eq(var a, var b)
bool _Ne(var a, var b)
bool _Lt(var a, var b)
bool _Le(var a, var b)
bool _Gt(var a, var b)
bool _Ge(var a, var b)

// These are explicit function names for primitive types
// These methods must also be explicitly listed in the
// _primitive_method_names dictionary above, and of course
// actually implemented in the C code.
// TODO: Implement a more elegant mechanism for enumerating
// primitive type methods.
String null_mRepr(var x) {
  return 'null'
}
bool null_mBool(var x) {
  return false
}
bool null_mEq(var a, var b) {
  return a is b
}

String bool_mRepr(bool b) {
  return b ? 'true' : 'false'
}
bool bool_mBool(bool b) {
  return b;
}
bool bool_mEq(bool a, bool b) {
  return a is b;
}

int int_mAdd(int a, int b)
int int_mSub(int a, int b)
int int_mMul(int a, int b)
int int_mDiv(int a, int b)
int int_mMod(int a, int b)
bool int_mEq(int a, int b)
bool int_mLt(int a, int b)
String int_mRepr(int a)
bool int_mBool(int x) {
  return x != 0
}

double double_mAdd(double a, double b)
double double_mSub(double a, double b)
double double_mMul(double a, double b)
double double_mDiv(double a, double b)
bool double_mEq(double a, double b)
bool double_mLt(double a, double b)
String double_mRepr(double a)
bool double_mBool(double x) {
  return x != 0.0
}

String function_mGETname(function f)
String function_mRepr(function f) {
  return '<function ' + f.name + '>'
}
bool function_mBool(function f) {
  return true
}

String type_mGETname(type t)
String type_mRepr(type t) {
  return '<type ' + t.name + '>'
}
bool type_mBool(type t) {
  return true
}
bool type_mEq(type a, type b)

trait Object() {
  bool Eq(var other) {
    return this is other
  }
  String Repr() {
    type t = type(this)
    return '<' + t.name + ' instance>'
  }
  String Str() {
    return this.Repr().(String)
  }
  bool Bool() {
    return true
  }
}

extern class WeakReference {
  new(var obj)
  var getNullable()

  bool Bool() {
    return this.getNullable() is not null
  }

  var get() {
    var obj = this.getNullable()
    if (obj is null) {
      panic("Get null weak reference")
    }
    return obj
  }
}

extern class String {
  int GETbytesize()
  String Add(String b)
  int GETsize()
  String Str()
  String Repr() {
    return '"' + this.escape() + '"'
  }
  String escape()
  bool Eq(String other)
  bool Lt(String other)
  bool Bool() {
    return bool(this.size)
  }
  String Slice(int start, int end)

  String SliceLeft(int start) {
    return this.Slice(start, this.size)
  }

  String SliceRight(int end) {
    return this.Slice(0, end)
  }

  String GetItem(int i) {
    if (i < 0) {
      i = i + this.size
    }
    return this.Slice(i, i + 1)
  }
}

extern class StringBuilder {
  new()
  void addstr(String s)

  // NOTE: The 'build' method empties out the StringBuilder
  String build()

  void add(var x) {
    this.addstr(str(x))
  }
}

extern class File {
  new(String path, String mode)
  void close()
  void write(String s)
  String read()
  void print(var x) {
    this.write(str(x))
    this.write('\n')
  }
}

extern class Closure {
  new(List captures, function f)
  extern Call
}

extern File STDIN
extern File STDOUT
extern File STDERR

// argc and argv passed to main
// made available as a List
extern List ARGS

void panic(String message)
bool bool(var x)
type type(var x)
void assert(var x)

void print(var x) {
  STDOUT.print(x)
}

String str(var x) {
  return x.Str().(String)
}

String repr(var x) {
  return x.Repr().(String)
}

trait Iterable {
  void forEach(var f) {
    final i = this.Iterator()
    while (i.HasNext()) {
      f(i.Next())
    }
  }
}

extern class List(Iterable) {
  int GETsize()
  var GetItem(int i)
  var SetItem(int i, var x)
  void push(var x)

  String Repr() {
    StringBuilder sb = StringBuilder()
    sb.addstr('[')
    int size = this.size
    int i = 0
    while (i < size) {
      if (i != 0) {
        sb.addstr(', ')
      }
      sb.addstr(repr(this[i]))
      i = i + 1
    }
    sb.addstr(']')
    return sb.build()
  }

  bool Eq(List other) {
    int size = this.size
    if (size != other.size) {
      return false
    }
    for (int i = 0; i < size; i++) {
      if (this[i] != other[i]) {
        return false
      }
    }
    return true
  }

  bool Contains(var x) {
    int size = this.size
    for (int i = 0; i < size; i++) {
      if (this[i] == x) {
        return true
      }
    }
    return false
  }

  ListIterator Iterator() {
    return ListIterator(this)
  }
}

class ListIterator {
  List list
  int i

  new(List list) {
    this.list = list
    this.i = 0
  }

  bool HasNext() {
    return this.i < this.list.size
  }

  var Next() {
    var value = this.list[this.i]
    this.i = this.i + 1
    return value
  }
}
