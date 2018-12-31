import klx.g.lib


class Impl {
  static Impl new() = $(Impl)

  var draw(var ctx, List rect) {
    print('draw rect = ' + str(rect))
    ctx.color = [100, 255, 100]
    ctx.rect(rect)
  }
}

void main() {
  var impl = Impl()
  lib.KLC_cocoa_sample_func(impl)
}
