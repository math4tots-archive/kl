
case class Foo {
  int a
  String s
}

void main() {
  final x = Foo(10, 'hi')
  final y = Foo(10, 'hi')
  final z = Foo(81, 'hi')
  print(x)
  assert(x == y)
  assert(x != z)
}
