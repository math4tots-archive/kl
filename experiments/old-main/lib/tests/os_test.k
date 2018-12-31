import os
import os.fs

void main() {
  if (false) {
    with open('foo.txt', 'r') as f {
      print(f.read())
    }
    print(os.name)
    final p = fs.join(['.', 'c'])
    print(fs.listdir(p))
    print(fs.tryListdir(p))
    print(fs.tryListdir(fs.join(['.', 'hi'])))
  }
}
