import os

void main() {
  print(os.getcwd())
  print(os.joinpath([os.getcwd(), '..']))

  final start = os.getcwd()
  for (int i = 0; i < 1000; i++) {
    os.mkdir('foo')
    os.chdir('foo')
    print('i = ' + str(i) + ', ' + str(os.getcwd().size) + ', ' + os.getcwd())
  }
  print('done!')
}
