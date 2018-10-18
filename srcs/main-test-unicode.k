void main() {
  final s = '世界'
  print('s = ' + str(s))

  // With UTF-8 these characters would be 3 bytes each,
  assert(s.bytesize == 6)

  // But there are really just 2 unicode code points.
  // assert(s.size == 2)
}
