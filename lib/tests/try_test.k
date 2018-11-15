try int foo(int x) {
  return x + 89041
}

class Foo {
  try String foo() {
    return 'hello foo'
  }
}

try int bar(int x) {
  raise x
}

void main() {
  assert(foo(0) == 89041)
  Try result = try.foo(0)
  assert(result)
  assert(result.orDie() == 89041)

  assert(Foo().foo() == 'hello foo')
  assert(type(Foo().try.foo()) == Try)
  assert(Foo().try.foo())
  assert(Foo().try.foo().orDie() == 'hello foo')

  assert(!try.bar(489))
  assert(try.bar(489).getRawValue() == 489)
}
