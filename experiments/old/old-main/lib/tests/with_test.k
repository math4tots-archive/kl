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

  checkReferenceExample()
}

void returnInsideWith(ContextManager cm) {
  with cm {
    return
  }
  assert(false)
}

class SomeClass {}

class ReferenceCheckingContextManager {
  WeakReference wr
  bool entered
  bool exited

  new() {
    this.entered = false
    this.exited = false
  }

  SomeClass Enter() {
    SomeClass ret = SomeClass()
    this.wr = WeakReference(ret)
    // Here we hold on to a strong reference (ret),
    // so the reference should still be alive.
    assert(this.wr)
    this.entered = true
    return ret
  }

  void Exit() {
    // Here, we're outside the 'with' block,
    // and we're checking that no one else holds
    // a reference this object anymore.
    assert(!this.wr)
    this.exited = true
  }
}

void checkReferenceExample() {
  final cm = ReferenceCheckingContextManager()
  assert(!cm.entered)
  assert(!cm.exited)
  with cm as sc {
    assert(type(sc) == SomeClass)
  }
  assert(cm.entered)
  assert(cm.exited)
}