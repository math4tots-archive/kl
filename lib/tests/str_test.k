void main() {
  final str = 'hello'
  final buf = str.encode('utf8')
  assert(buf.dumpstr() == '<Buffer 104, 101, 108, 108, 111>')
  assert(buf == BufferFromByteList([
    104, 101, 108, 108, 111
  ]))
  assert(buf.decode('utf8') == 'hello')
}
