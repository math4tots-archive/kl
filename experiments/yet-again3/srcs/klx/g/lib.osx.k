#include "klx/g/lib.osx.h"
#source "lib.m"
#framework "Cocoa"

class Ref {
  c_int id

  static Ref new(c_int id) {
    Ref ref = $(Ref)
    ref->id = id
    ref
  }

  delete {
    KLC_cocoa_release(id)
  }
}


extern void KLC_cocoa_sample_func(var impl)

extern var KLC_cocoa_int_to_var(c_int id) = Ref(id)
extern void KLC_cocoa_release(c_int id)


class DrawingContext {

  List __SETcolor(List color) {
    assert.equal(color.size, 3)
    KLC_cocoa_set_color(color[0], color[1], color[2])
    color
  }

  void rect(List rect) {
    assert.equal(rect.size, 4)
    KLC_cocoa_fill_rect(rect[0], rect[1], rect[2], rect[3])
  }
}

extern var KLC_cocoa_drawing_context() = $(DrawingContext)

extern void KLC_cocoa_fill_rect(float x, float y, float w, float h)
extern void KLC_cocoa_set_color(int r, int g, int b)
