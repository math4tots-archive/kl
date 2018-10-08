
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

String bool_mRepr(bool b) {
  return b ? 'true' : 'false'
}
bool bool_mBool(bool b) {
  return b;
}

int int_mAdd(int a, int b)
int int_mSub(int a, int b)
bool int_mEq(int a, int b)
bool int_mNe(int a, int b)
bool int_mLt(int a, int b)
String int_mRepr(int a)
bool int_mBool(int x) {
  return x != 0
}

double double_mAdd(double a, double b)
double double_mSub(double a, double b)
bool double_mEq(double a, double b)
bool double_mNe(double a, double b)
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
bool type_mBool(function f) {
  return true
}

trait Object() {
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
  bool Bool() {
    return bool(this.size)
  }
}

bool bool(var x)
type type(var x)
void puts(String s)

void print(var x) {
  puts(str(x))
}

String str(var x) {
  return x.Str().(String)
}

String repr(var x) {
  return x.Repr().(String)
}
