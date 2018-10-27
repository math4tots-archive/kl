import os
import os.path as pt

void main() {
  if (false) {
    with open('foo.txt', 'r') as f {
      print(f.read())
    }
    print(os.name)
    final p = pt.join(['.', 'c'])
    print(os.listdir(p))
    print(os.tryListdir(p))
    print(os.tryListdir(pt.join(['.', 'hi'])))
  }
}
