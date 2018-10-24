// Test that things are hashable

void test() {
  final s = {}
  assert(type(s) is Set)
  s.add(10)
  assert(10 in s)
  assert(10.0 in s)
  assert(5 not in s)
  assert(s.size == 1)
  s.add('hi')
  assert('hi' in s)
  assert(s.size == 2)

  s.add(10.0)
  assert(10 in s)
  assert(10.0 in s)
  assert(10.1 not in s)
  assert(s.size == 2)

  s.add(null)
  assert(s.size == 3)
  assert(null in s)

  assert(test not in s)
  s.add(test)
  assert(s.size == 4)
  assert(test in s)

  assert(Set not in s)
  s.add(Set)
  assert(s.size == 5)
  assert(Set in s)
  assert(10 in s)

  // Test hash collision
  {
    assert('hi'.HashCode() == 'hi'.HashCode().HashCode())
    final x = Set()
    x.add('hi')
    assert('hi'.HashCode() not in x)
    x.add('hi'.HashCode())
    assert(x.size == 2)
    assert('hi' in x)
    assert('hi'.HashCode() in x)
  }
}
