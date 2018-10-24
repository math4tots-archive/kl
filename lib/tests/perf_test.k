// Test that things are hashable

int transform(int i) {
  return i % 2 == 0 ? i / 2 : (3 * i + 1)
}

void test() {
  final xs = []
  for (int i = 0; i < 10 ** 6; i++) {
    xs.push(i)
  }
  print('xs.size = ' + str(xs.size))
}
