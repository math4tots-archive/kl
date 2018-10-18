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

  assert(s.Slice(1, 2) == 'e')
  assert(s.Slice(0, 5) == 'Hello')
  assert('Hello,'.Slice(0, 5) == 'Hello')
  assert(s.Slice(7, 8) == '世')
  assert(s.Slice(7, 9) == '世界')
  assert(s.SliceLeft(7) == '世界')
  assert(s[0] == 'H')
  assert(s[s.size - 1] == '界')
  assert(s[-1] == '界')
}
