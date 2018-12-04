from c import printf

trait SomeTrait() {
  static int someStaticMethod() {
    printf("Inside someStaticMethod\n")
  }

  int someInstanceMethod() {
    printf("Inside someInstanceMethod\n")
  }
}

trait SomeOtherTrait {
  void someOtherInstanceMethod() {
    printf("Inside someOtherInstanceMethod\n")
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
  printf("list.size = %ld\n", size)
  list.push(15)
  size = list.size
  printf("list.size = %ld\n", size)

  SomeTrait.someStaticMethod()

  SomeClass someClass = SomeClass()
  someClass.someInstanceMethod()
  someClass.someOtherInstanceMethod()
  someClass.someInstanceMethodOnObject()

  print(someClass)

  ${
    type t = SomeOtherTrait
    var tv = t
    t = tv
    print(t)
  }

  static_cast(c_int, 0)
}
