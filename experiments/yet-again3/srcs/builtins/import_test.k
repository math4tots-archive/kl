import builtins.testmodulea as t
from builtins.testmoduleb import bar

void main() {
  assert.equal(t.foo(), 1042)
  assert.equal(bar(), 'result of bar')
}
