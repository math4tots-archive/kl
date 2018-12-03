from c import printf

trait SomeTrait {
  static int someStaticMethod() {
    printf("Inside someStaticMethod\n")
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

  static_cast(c_int, 0)
}
