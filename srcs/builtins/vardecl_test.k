
void used_before_declared_int() {
  // Unfortunately, a variable can be used in a scope in which it is
  // defined, before its actual definition.
  // We might want to prevent this in the future, but at least for now
  // make sure that we get consistent initial values.
  assert.equal(x, 0)
  int x = 10
  assert.equal(x, 10)
}


void used_before_declared_str() {
  // print(s)  // this should throw
  String s = 'hi'
  // String s  // this should throw error at compile time (must initialize)
}


void main() {
  used_before_declared_int()
  used_before_declared_str()
}
