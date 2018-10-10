
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
bool int_mEq(int a, int b)
bool int_mLt(int a, int b)
String int_mRepr(int a)
bool int_mBool(int x) {
  return x != 0
}

double double_mAdd(double a, double b)
double double_mSub(double a, double b)
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
}

extern class String {
  int bytesize()
  String Add(String b)
  int GETsize()
  String Str()
  bool Eq(String other)
  bool Lt(String other)
  bool Bool() {
    return bool(this.size)
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

bool bool(var x)
type type(var x)
void puts(String s)
void assert(var x)

void print(var x) {
  puts(str(x))
}

String str(var x) {
  return x.Str().(String)
}

String repr(var x) {
  return x.Repr().(String)
}

extern class List {
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
}

