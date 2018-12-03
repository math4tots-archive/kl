from c import printf

extern c_int main() {
  List list = List()
  c_long size = list.size
  printf("list.size = %ld\n", size)
  list.push(15)
  size = list.size
  printf("list.size = %ld\n", size)
  static_cast(c_int, 0)
}
