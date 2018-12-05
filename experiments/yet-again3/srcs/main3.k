from c import printf

trait SomeTrait() {
  static int someStaticMethod() {
    printf(c"Inside someStaticMethod\n")
  }

  int someInstanceMethod() {
    printf(c"Inside someInstanceMethod\n")
  }
}

trait SomeOtherTrait {
  void someOtherInstanceMethod() {
    printf(c"Inside someOtherInstanceMethod\n")
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
  printf(c"list.size = %ld\n", size)
  list.push(15)
  size = list.size
  printf(c"list.size = %ld\n", size)

  SomeTrait.someStaticMethod()

  SomeClass someClass = SomeClass()
  someClass.someInstanceMethod()
  someClass.someOtherInstanceMethod()
  someClass.someInstanceMethodOnObject()

  print(someClass)

  // ${
  //   int i = 0
  //   while (i < 10) {
  //     print("i = " + str(i))
  //   }
  // }

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
