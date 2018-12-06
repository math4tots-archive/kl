from builtins import KLC_release_vars_queued_for_exit
from c import printf

trait SomeTrait() {
  static void someStaticMethod() {
    print("Inside someStaticMethod")
  }

  void someInstanceMethod() {
    print("Inside someInstanceMethod")
  }
}

trait SomeOtherTrait {
  void someOtherInstanceMethod() {
    print("Inside someOtherInstanceMethod")
  }
}

class SomeClass(SomeTrait, SomeOtherTrait) {
  static SomeClass new() = $(SomeClass)
}

String message = 'hello world'
List items = []

extern c_int main() {
  List list = List()
  c_long size = list.size
  print('list.size = ' + str(size))
  list.push(15)
  size = list.size
  print('list.size = ' + str(size))

  SomeTrait.someStaticMethod()

  SomeClass someClass = SomeClass()
  someClass.someInstanceMethod()
  someClass.someOtherInstanceMethod()
  someClass.someInstanceMethodOnObject()

  print(someClass)

  print([1, 2, 3])
  print('[1, 2, 3].size = ' + str([1, 2, 3].size))

  print('(global) message = ' + message)
  print('(global) items = ' + str(items))
  items.push(1)
  items.push(2)
  items.push(3)
  print('(global) items = ' + str(items))

  // ${
  //   int i = 0
  //   int total = 0
  //   while (i < 1000000000) {
  //     total = total + i
  //     i = i + 1
  //   }
  //   print('total = ' + str(total))
  // }

  ${
    var a = 10
    var b = 5
    printf(c'a is b = %d\n', static_cast(c_int, a is b))
    printf(c'a is not b = %d\n', static_cast(c_int, a is not b))
    printf(c'a is b = %d\n', static_cast(c_int, a is a))
  }

  ${
    print('a')
    int i = 0
    while (i < 10) {
      print("i = " + str(i))
      i = i + 1
    }
  }

  ${
    type t = SomeOtherTrait
    var tv = t
    t = tv
    print(t)
  }

  ${
    StringBuilder sb = StringBuilder()
    sb.addString(String.fromCString(c"abc"))
    sb.addString(String.fromCString(c"def"))
    print(sb)
  }

  KLC_release_vars_queued_for_exit()
  static_cast(c_int, 0)
}
