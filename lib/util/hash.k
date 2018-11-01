

int list(List xs) {
  // Hash a list of elements
  // From openjdk Java's Arrays.hashCode(int[])

  int result = 1
  for (var element: xs) {
    result = 31 * result + hash(element)
  }
  return result
}
