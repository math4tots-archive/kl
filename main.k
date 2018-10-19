#include "os"
#include "posix"

void main() {
  print(OS.getcwd())
  print(OS.joinpath([OS.getcwd(), '..']))

  final start = OS.getcwd()
  for (int i = 0; i < 1000; i++) {
    OS.mkdir('foo')
    OS.chdir('foo')
    print('i = ' + str(i) + ', ' + str(OS.getcwd().size) + ', ' + OS.getcwd())
  }
  print('done!')
}
