void main() {
  assert.equal(
    range(0, 10).to(List),
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  )

  var xs = [1, 2, 3]

  assert.equal(xs[0], 1)
  assert.equal(xs[-1], 3)
  assert.equal(xs.size, 3)
}
