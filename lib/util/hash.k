

int list(List xs) {
  // Hash a list of elements
  // From openjdk Java's Arrays.hashCode(int[])

  int result = 1
  for element in xs {
    result = 31 * result + hash(element)
  }
  return result
}
