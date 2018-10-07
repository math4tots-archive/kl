
// These are explicit function names for primitive types
// These methods must also be explicitly listed in the
// _primitive_method_names dictionary above, and of course
// actually implemented in the C code.
// TODO: Implement a more elegant mechanism for enumerating
// primitive type methods.
String null_mRepr(var x) {
  return 'null'
}

String bool_mRepr(bool b) {
  return b ? 'true' : 'false'
}

int int_mAdd(int a, int b)
int int_mSub(int a, int b)
bool int_mEq(int a, int b)
bool int_mLt(int a, int b)
String int_mRepr(int a)

String double_mRepr(double d)

String function_mGETname(function f)
String function_mRepr(function f) {
  return '<function ' + f.name + '>'
}

String type_mGETname(type t)
String type_mRepr(type t) {
  return '<type ' + t.name + '>'
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
