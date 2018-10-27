// Test that 'with' statements work as expected

class ContextManager {
  bool exited

  new() {
    this.exited = false
  }

  int Enter() {
    return 49
  }

  void Exit() {
    this.exited = true
  }
}

void main() {
  final cm = ContextManager()

  with cm as x {
    assert(x == 49)
    assert(!cm.exited)
  }
  assert(cm.exited)

  // Check that even if you return inside a with statement,
  // the 'Exit' method is properly called
  cm.exited = false
  returnInsideWith(cm)
  assert(cm.exited)
}


void returnInsideWith(ContextManager cm) {
  with cm {
    return
  }
  assert(false)
}
