#include "posix"

void main() {
  assert(POSIX)
  assert(POSIX.VERSION >= 200112)

  print(POSIX)
  print(POSIX.getcwd())
  print(POSIX.readdir('.'))
  print(POSIX.readdir('..'))
  print(POSIX.listdir('.'))
  print('POSIX.VERSION = ' + str(POSIX.VERSION))
}
