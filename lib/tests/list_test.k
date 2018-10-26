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
}
