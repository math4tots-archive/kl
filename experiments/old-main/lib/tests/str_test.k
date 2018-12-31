void main() {
  final str = 'hello'
  final buf = str.encode('utf8')
  assert(buf.dumpstr() == '<Buffer 104, 101, 108, 108, 111>')
  assert(buf == BufferFromByteList([
    104, 101, 108, 108, 111
  ]))
  assert(buf.decode('utf8') == 'hello')
  startsWith()
}


void startsWith() {
  assert(ord('a') + 1 == ord('b'))
  assert(ord('a') + 2 == ord('c'))
  assert(ord('a') + 3 == ord('d'))
  assert(ord('a') + 4 == ord('e'))
  assert(ord('a') + 5 == ord('f'))
  assert(ord('A') + 1 == ord('B'))
  assert(ord('a') != ord('A'))
  assert(ord('A') - ord('a') == ord('B') - ord('b'))
  assert(ord('世') == 19990)
  assert(ord('界') == 30028)
  for (int i = 0; i < 10; i++) {
    assert(ord('0') + i == ord(str(i)))
  }
  assert('hello'.startsWith('h'))
  assert('hello'.startsWith('he'))
  assert(!'hello'.startsWith('hx'))

  assert(!'hello'.startsWithAt('h', 1))
  assert('hello'.startsWithAt('e', 1))
  assert('hello'.startsWithAt('ello', 1))
  assert(!'hello'.startsWithAt('ellox', 1))
}
