
void main() {
  int n = 50

  for (int i = 1; i <= 50; i = i + 1) {
    if (i % 3 == 0 and i % 5 == 0) {
      print('fizzbuzz')
    } else if (i % 3 == 0) {
      print('fizz')
    } else if (i % 5 == 0) {
      print('buzz')
    } else {
      print(i)
    }
  }
}
