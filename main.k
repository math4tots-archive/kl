#include "os"
#include "posix"

void main() {
  print(OS.getcwd())
  print(OS.joinpath([OS.getcwd(), '..']))
}
