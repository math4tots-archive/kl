
case class Foo {
  int a
  String s
}

case class Empty {}

void main() {
  final x = Foo(10, 'hi')
  final y = Foo(10, 'hi')
  final z = Foo(81, 'hi')
  assert(str(x) == 'tests.case_class_test.Foo(10, "hi")')
  assert(x == y)
  assert(x != z)
  assert(x.GetFields() == [10, 'hi'])

  final set = {x, y, z}
  assert(set == {Foo(10, 'hi'), Foo(81, 'hi')})
  assert(sorted(set) == [Foo(10, 'hi'), Foo(81, 'hi')])

  assert(sorted([
    Foo(2, 'a'), Foo(1, 'b'), Foo(1, 'd'), Foo(1, 'a'), Foo(1, 'c'),
  ]) == [
    Foo(1, 'a'), Foo(1, 'b'), Foo(1, 'c'), Foo(1, 'd'), Foo(2, 'a'),
  ])

  assert(Empty() == Empty())
  assert(str(Empty()) == 'tests.case_class_test.Empty()')
  assert(Empty().GetFields() == [])
}
