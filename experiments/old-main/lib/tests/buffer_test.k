void main() {
  misc()
  resize()
  view()
}

void misc() {
  final b = Buffer(8)
  assert(b.dumpstr() == '<Buffer 0, 0, 0, 0, 0, 0, 0, 0>')
  assert(b.get4(0) == 0)
  b.set4(0, 15)
  assert(b.get4(0) == 15)
  assert(b.dumpstr() == '<Buffer 15, 0, 0, 0, 0, 0, 0, 0>')
  b.set4(0, 4000000000)
  assert(b.get4(0) == 4000000000)
  assert(b.dumpstr() == '<Buffer 0, 40, 107, 238, 0, 0, 0, 0>')
  b.set4(4, 4000000000)
  assert(b.get4(0) == 4000000000)
  assert(b.get4(4) == 4000000000)
  assert(b.get1(0) == 0)
  assert(b.get1(1) == 40)
  assert(b.get1(2) == 107)
  assert(b.get1(3) == 238)
  assert(b.dumpstr() == '<Buffer 0, 40, 107, 238, 0, 40, 107, 238>')
  assert(b == BufferFromByteList([
    0, 40, 107, 238, 0, 40, 107, 238
  ]))
}

void resize() {
  final b = Buffer(8)
  assert(b.size == 8)
  assert(b == BufferFromByteList([0] * 8))
  b.resize(0)
  assert(b.size == 0)
  assert(b == BufferFromByteList([]))
  b.resize(16)
  assert(b == BufferFromByteList([0] * 16))
}

void view() {
  final b = Buffer(8)
  final v = b.view(1, 3)
  assert(v.get1(0) == 0)
  b.set1(1, 15)
  assert(v.get1(0) == 15)
}
