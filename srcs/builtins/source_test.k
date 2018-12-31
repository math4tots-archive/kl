#include "builtins/source_test.h"
#source "source_test.c"

extern c_int KLC_builtins_source_test_foo()


void main() {
  assert.equal(KLC_builtins_source_test_foo(), 10985)
}
