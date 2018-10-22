import os
import os.path as pt

void main() {
  print(os.name)
  final p = pt.join(['.', 'c'])
  print(os.listdir(p))
  print(os.tryListdir(p))
  print(os.tryListdir(pt.join(['.', 'hi'])))
}
