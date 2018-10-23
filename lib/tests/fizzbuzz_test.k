List fizzbuzz(int n) {
  final ret = []

  for (int i = 1; i <= n; i = i + 1) {
    if (i % 3 == 0 and i % 5 == 0) {
      ret.push('fizzbuzz')
    } else if (i % 3 == 0) {
      ret.push('fizz')
    } else if (i % 5 == 0) {
      ret.push('buzz')
    } else {
      ret.push(i)
    }
  }

  return ret
}

void test() {
  assert(fizzbuzz(50) == [
    1,
    2,
    "fizz",
    4,
    "buzz",
    "fizz",
    7,
    8,
    "fizz",
    "buzz",
    11,
    "fizz",
    13,
    14,
    "fizzbuzz",
    16,
    17,
    "fizz",
    19,
    "buzz",
    "fizz",
    22,
    23,
    "fizz",
    "buzz",
    26,
    "fizz",
    28,
    29,
    "fizzbuzz",
    31,
    32,
    "fizz",
    34,
    "buzz",
    "fizz",
    37,
    38,
    "fizz",
    "buzz",
    41,
    "fizz",
    43,
    44,
    "fizzbuzz",
    46,
    47,
    "fizz",
    49,
    "buzz",
  ])
}
