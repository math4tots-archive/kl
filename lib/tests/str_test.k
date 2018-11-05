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
  assert('hello'.startsWith('h'))
  assert('hello'.startsWith('he'))
  assert(!'hello'.startsWith('hx'))

  assert(!'hello'.startsWithAt('h', 1))
  assert('hello'.startsWithAt('e', 1))
  assert('hello'.startsWithAt('ello', 1))
  assert(!'hello'.startsWithAt('ellox', 1))
}
