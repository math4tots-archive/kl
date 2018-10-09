#include "./a.k"
#include "./b.k"

class Bar {
  double d
  String s

  int f() {
    return 10
  }
}

class Baz {
  new(int a) {
    print('Baz.a')
  }
}

int ff() {
  return 184
}

String _f() {
  return 'result of _f'
}

String sample_global_string = """
This is a sample global string
"""

void main() {
  assert(true)
  assert(!false)

  assert('Hello world!' == 'Hello world!')
  assert('Hello world!'.size == 12)

  {
    // primitive comparison
    assert(1 < 10)
    assert(10 > 1)
    assert(-5 <= 0)
    assert(0 >= 0)
    assert(!(10 < -1))
  }

  {
    // String comparison
    assert('abc' == 'abc')
    assert('abc' != 'deff')
    assert(!('abc' == 'def'))
    assert('abc' < 'def')
    assert(!('abc' > 'def'))
    assert('abc' <= 'abc')
    assert('abc' >= 'abc')
  }

  {
    // Mixed type comparisons
    // NOTE: mixed type comparisons incur an overhead as
    // a call to a generic function is made
    var a = 10
    assert(a == 10)
    assert(a == 10.0)  // special case for int,double comparison
    assert(10 == 10.0)
    assert(10.0 == 10)
    assert(9.5 < 10)
    assert('hello world' != 15)

    assert(17.0 == 17)
    assert(17.0 is not 17)
    assert(!(17.0 is 17))
  }

  {
    var i = 81
    assert(i == 81)
    assert(10 + 15 == 25)
  }

  {
    Bar bar = Bar()
    assert(bar.f() == 10)
    assert(str(bar) == '<Bar instance>')
  }

  {
    String s = 'original s variable'
    assert(s == 'original s variable')
    s = 'new value'
    assert(s == 'new value')
  }

  {
    assert([1, 2, 3].size == 3)
  }

  {
    assert(String == String)
  }

  oldmain()
}

void oldmain() {
  String s = 'String s variable'
  var v = 'var v variable'
  print('Hello world!')
  print('Hello world!'.size)
  print(s)
  print(s.size)
  print(v)
  print(v.size)
  print(ff())
  print(15 + 17)
  var i = 81
  print(i + 5)
  print(81 + 5)
  print(i - 5)
  print(81 == 81)
  print(_f())
  Bar bar = Bar()
  print(bar)
  Baz baz = Baz(15)
  print(baz)

  bar.d = 10.
  print('bard.d = ' + str(bar.d))

  print(bar.s)
  bar.s = 'bar.s content'
  print(bar.s)

  {
    String vs = v.(String)
    print('vs = ' + vs)
  }

  if (1 < 10) {
    print('1 < 10 is true')
  } else if (1 == 10) {
    print('1 == 10 10 is true')
  } else {
    print('1 < 10 is false, or something')
  }

  v = '"result-of-new-v"'
  print('v = ' + str(v))

  {
    int j = 0
    while (j < 10) {
      print('j = ' + str(j))
      j = j + 1
    }
  }

  {
    String ss = 'hi'
    ss = 'there'
  }

  print(1 || 2)
  print(1 && 2)
  print(0 && 1)
  print(0 || 1)
  print(0.0 || 0)
  print('' && '')
  print('a' || '')
  print(0 ? 1 : 2)
  print(true ? 1 : 2)
  print(null)
  print(sample_global_string)
  print(sample_global_string)
  print(sample_global_string)
  print(sample_global_string)
  print(!true)
  print(!0)
}
