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
  static SomeClass new() {
    $(SomeClass)
  }
}

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

  static_cast(c_int, 0)
}
