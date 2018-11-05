void main() {
  {
    final xs = []
    for (int i = 0; i < 10 ** 6; i++) {
      xs.push(i)
    }
    assert(xs.size == 10 ** 6)
    for (int i = 0; i < 10 ** 6; i++) {
      xs.pop()
    }
    assert(xs.size == 0)
    assert(!xs)
  }

  {
    auto s = 'hi'
    final wr = WeakReference(s)
    final xs = [s]
    assert(wr.refcnt == 2)
    s = null(String)
    assert(wr.refcnt == 1)
    xs.resize(4)
    assert(xs == ['hi', null, null, null])
    assert(wr.refcnt == 1)
    xs.clear()
    assert(xs == [])
    assert(wr.refcnt == 0)
  }

  sorts()
  assert(sorted([3, 2, 1]) == [1, 2, 3])

  rev()
}

void sorts() {
  final xs = [1, 85, 29, 9, 841, 1, 2]
  xs.sort()
  assert(xs == [1, 1, 2, 9, 29, 85, 841])

  final ss = ['a', 'b', 'ab', 'd', 'd', 'eb', 'easdf', 'c']
  ss.sort()
  assert(ss == ["a", "ab", "b", "c", "d", "d", "easdf", "eb"])
}

void rev() {
  assert([1, 2, 3] == reversed([3, 2, 1]))
  assert([] == reversed([]))
  assert([19] == reversed([19]))
}
