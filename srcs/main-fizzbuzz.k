
void main() {
  int n = 50
  int i = 1

  while (i <= 50) {
    if (i % 3 == 0 and i % 5 == 0) {
      print('fizzbuzz')
    } else if (i % 3 == 0) {
      print('fizz')
    } else if (i % 5 == 0) {
      print('buzz')
    } else {
      print(i)
    }
    i = i + 1
  }
}
