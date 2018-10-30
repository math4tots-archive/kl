void main() {
  bitwise()
}

void bitwise() {
  assert(3 | 5 == 7)
  assert(3 & 5 == 1)
  assert(3 ^ 5 == 6)
  assert(1 << 10 == 1024)
  assert(1024 >> 2 == 256)

  // I'm not sure if this is standard, but
  // I want to test for consistent behavior.
  assert(-1 >> 1 == -1)
  assert(-2 >> 1 == -1)
  assert(-2 << 1 == -4)
}
