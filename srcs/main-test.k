#include "./a"
#include "./b"
#include "os"

class Bar {
  double d
  String s

  int f() {
    return 10
  }
}

class Baz {
  int a

  new(int a) {
    this.a = a
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

class F0 {
  String s

  new(String s) {
    this.s = s
  }

  String Call() {
    return 'xx' + this.s + 'xx'
  }
}

String f0() {
  return 'hello'
}

String f1(function f) {
  return f().(String) + ' world'
}

String f2(var f) {
  return f().(String) + ' world'
}

String f3(F0 f) {
  return f() + ' world'
}

void main() {
  {
    assert(true)
    assert(!false)
    assert(!0 is true)
    assert((true or false) is true)
    assert((false or false) is false)
    assert((true and false) is false)
    assert((false and false) is false)
    assert((true and true) is true)

    assert((1 or 2) is true)
    assert((1 and 2) is true)
    assert((0 and 1) is false)
    assert((0 or 1) is true)
    assert((0.0 or 0) is false)
    assert(('' and '') is false)
    assert(('a' or '') is true)
    assert((0 ? 3 : 4) is 4)
    assert((true ? 3 : 4) is 3)
    assert(str(null) == 'null')
  }

  {
    // Arithmetic and function calls
    var i = 81
    assert(i == 81)
    assert(10 + 15 == 25)
    assert(ff() == 184)

    assert(15 + 17 == 32)
    assert(i + 5 == 86)
    assert(81 + 5 == 86)
    assert(i - 5 == 76)
    assert(81 == 81)
    assert(_f() == 'result of _f')
  }

  {
    assert('Hello world!' == 'Hello world!')
    assert('Hello world!'.size == 12)

    String s = 'Hello world!'
    assert(s == 'Hello world!')
    assert('Hello world!' == s)
    assert(s.size == 12)

    var v = 'Hello world!'
    assert(v == 'Hello world!')
    assert('Hello world!' == v)
    assert(v.size == 12)

    String vs = v.(String)
    assert(vs == s)
    assert(vs == v)
    assert('vs = ' + vs == 'vs = Hello world!')

    v = '"result-of-new-v"'
    assert(str(v) == v)
    assert(v == '"result-of-new-v"')

    s = 'new s value'
    assert(s != 'Hello world!')
    assert(s == 'new s value')
  }

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

    // Equality test with null
    assert('' != null)
    String s1 = ''
    String s2
    assert(s1 != s2)
    assert(null == null)
    assert(null(String) == null(Bar))
    assert(null(Bar) == s2)
    assert(s2 == null(Bar))
    assert(s1 != null(Bar))
    assert(null(String) != s1)
  }

  {
    Bar bar = Bar()
    assert(bar.s is null)
    assert(bar.s == null)
    bar.s = 'bar.s content'
    assert(bar.s == 'bar.s content')
    assert(bar.f() == 10)
    assert(str(bar) == '<Bar instance>')

    Baz baz = Baz(15)
    assert(str(baz) == '<Baz instance>')
    assert(baz.a == 15)

    bar.d = 10.
    assert(bar.d is 10.0)
    assert(bar.d == 10)
    assert(bar.d is not 10)
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

  {
    StringBuilder sb = StringBuilder()
    assert(sb.build() == '')
    sb.add('hello world!')
    assert(sb.build() == 'hello world!')
    sb.add('part1 ')
    sb.add('part2')
    assert(sb.build() == 'part1 part2')
    sb.add('hello ')
    sb.add(15)
    assert(sb.build() == 'hello 15')
  }

  {
    assert(str([1, 2, 3, 4, 5]) == '[1, 2, 3, 4, 5]')
    assert(str([]) == '[]')
    assert(str([17]) == '[17]')

    List xs = []
    int j = 0
    while (j < 10) {
      xs.push(j)
      j = j + 1
    }
    assert(str(xs) == '[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]')
  }

  {
    String result
    if (1 < 10) {
      result = '1 < 10 is true'
    } else if (1 == 10) {
      result = '1 == 10 10 is true'
    } else {
      result = '1 < 10 is false, or something'
    }
    assert(result == '1 < 10 is true')
  }

  {
    assert(sample_global_string == '\nThis is a sample global string\n')
  }

  {
    // Weak reference test
    Bar b = Bar()
    final wr = WeakReference(b)
    assert(wr)
    assert(wr.get() is b)
    b = null(Bar)
    assert(!wr)
    assert(wr.getNullable() is null)
  }

  // truthy
  {
    String s

    // null variables, even if concretely typed,
    // should return false
    assert(!s)
  }

  // function as variable
  {
    assert(f1(f0) == 'hello world')
    assert(f2(f0) == 'hello world')
    assert(f1(f0) == f2(f0))
    final f = F0('hello')
    assert(f2(f) == f3(f))
    assert(f2(f) == 'xxhelloxx world')
  }

  // lambda expressions
  {
    final x = 10
    var addX = def int[int x](int y) {
      return x + y
    }
    assert(addX(15) == 25)
    assert(addX(17) == 27)
    assert(str(type(addX)) == '<type Closure>')

    // Check for leaks
    {
      String s = 'captured value'
      final wr = WeakReference(s)

      var lambda = def void[String s]() {}

      // There are two references to 'captured value',
      assert(wr)

      // Now we remove one of the references. But
      // the lambda should still be pointing to it.
      s = 'changed'
      assert(wr)

      // Now 'captured value' should have been collected
      lambda = null
      assert(!wr)
    }
  }

  // Test for loop, List.Contains, and Iterable protocol
  {
    final xs = []
    assert(3 not in xs)
    for (int i = 0; i < 5; i++) {
      xs.push(i)
    }
    assert(xs == [0, 1, 2, 3, 4])
    assert(3 in xs)
    assert(5 not in xs)
    assert(-1 not in xs)

    final ys = []
    xs.forEach(def void[List ys](int i) {
      ys.push(i)
    })
    assert(ys == [0, 1, 2, 3, 4])
    assert(xs == ys)
  }

  // Directory stuff
  {
    assert(OS.isdir('.'))
  }

  print(OS.name)
  print('OS.getcwd() = ' + str(OS.getcwd()))
  print('OS.posix = ' + str(OS.posix))
  print('OS.isdir("c/hw.c") = ' + str(OS.isdir("c/hw.c")))
  print('OS.isdir("srcs") = ' + str(OS.isdir("srcs")))
  print('OS.isfile("c/hw.c") = ' + str(OS.isfile("c/hw.c")))
  print('OS.isfile("srcs") = ' + str(OS.isfile("srcs")))
  print('ARGS = ' + str(ARGS))
  print(OS.listdir('.'))
  print('Hello world!')
  final data = File('c/hw.c', 'r').read()
  print('data.size = ' + str(data.size))
  print('After reading main.c')
}
