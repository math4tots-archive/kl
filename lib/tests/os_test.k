import os
import os.path as pt

void test() {
  if (false) {
    print(os.name)
    final p = pt.join(['.', 'c'])
    print(os.listdir(p))
    print(os.tryListdir(p))
    print(os.tryListdir(pt.join(['.', 'hi'])))
  }
}
