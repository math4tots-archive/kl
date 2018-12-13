// Some sanity checks

List a_global_list = []

trait SomeTrait {
  class String someInheritedClassMethod() = 'someInheritedClassMethod result'
}

class SomeClass(SomeTrait) {
  static SomeClass new() = $(SomeClass)

  class void foo(int x) {
    if (x < 10) {
      foo(x + 1)
    }
  }
}

class StringPointer {
  String s
}

void main() {
  assert(true)
  assert(!false)
  assert(!!true)
  assert.equal(1, 1)
  assert.equal(1, 1.0)
  assert(1 is not 1.0)
  assert.equal(1.0, 1.0)
  assert(1.1 != 1.0)
  assert('abc' == 'abc')
  assert('abcd' != 'abce')
  assert('abc' != 'abcd')
  assert('abc' != 3)
  assert(3 != 'abc')

  ${
    float f = if (1) {
      2
    } else {
      1.5
    }
    assert(f is 2.0)
    assert(f is not 2)
  }

  assert.equal(a_global_list, [])
  assert(a_global_list != [15])
  a_global_list.push(15)
  assert.equal(a_global_list, [15])
  assert(a_global_list != [16])

  assert.equal([1, 2, 3].map(x => x + 1), [2, 3, 4])
  ${
    var list = [1, 2, 3]
    list[0] = 15
    assert.equal(list, [15, 2, 3])
  }

  ${
    // Tests that assigning to a class field
    // properly increments reference count.
    StringPointer sp = $(StringPointer)
    sp->s = 'hi'
  }

  ${
    var iter = [1, 2, 3].__iter()
    var items = []
    while (iter.__next(x => items.push(x))) {}
    assert.equal(items, [1, 2, 3])
  }

  assert.equal([1, 2, 3, 4, 5].filter(x => x < 3), [1, 2])
  assert.equal([1, 2, 3, 4, 5].filter(x => x > 3), [4, 5])

  ${
    int i = 0
    var items = []
    while (i < 100) {
      items.push(str(i))
      i = i + 1
    }
    assert.equal(items.size, 100)
    assert.equal(items[0], '0')
    assert.equal(items[-1], '99')
    assert.equal(List.fromIterable(items), items)
    assert(List.fromIterable(items) is not items)
  }

  assert.equal(SomeClass.foo(1), null)

  // Test class methods
  assert.equal(
    SomeClass.someInheritedClassMethod(),
    'someInheritedClassMethod result')
  assert.equal(List.unit(10), [10])

  // Test lambda functions that return void
  ${
    var f = def void() {}
    assert.equal(f(), null)
  }

  // Test exponents
  assert.equal(2 ** 0, 1)
  assert(2 ** 0 is 1)
  assert.equal(2 ** 10, 1024)
  assert(2 ** 10 is not 1024.0)
  assert.equal(9 ** 0.5, 3)
  assert(2.0 ** 2 is 4.0)
  assert(2 ** 2 is 4)

  // Test streams
  ${
    assert.equal(range(10, 15).to(List), [10, 11, 12, 13, 14])
    assert.equal(
      range(0, 10 ** 9).take(5).to(List),
      [0, 1, 2, 3, 4],
    )
  }
}
