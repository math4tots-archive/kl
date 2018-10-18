#include "os"
#include "posix"

void main() {
  final s = "Hello, 世界"
  print(s)
  print(OS)
  print(POSIX)

  print('Before opening file')
  final f = File('foo.txt', 'w')
  f.print('asdfasdf')
  print('After printing ascii')
  f.print(s)
}
