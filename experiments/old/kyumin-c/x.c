#include <stdio.h>

int f(int x) {
  return x + 1;
}

int (*retfp())(int x) {
  return f;
}

int main() {
  printf("f() = %d\n", retfp()(10));
  return 0;
}
