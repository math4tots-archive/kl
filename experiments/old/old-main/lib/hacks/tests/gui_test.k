import hacks.gui


class State {
  int r
  int c
  int R
  int C
}

State state = State()

void redraw() {
  final windowSize = gui.getSize()
  final cellSize = gui.Size(
    windowSize.width / intToDouble(state.C),
    windowSize.height / intToDouble(state.R),
  )
  with gui.graphics() as gg {
    gg.setFillColor(gui.white)
    gg.fillBackground()

    gg.setFillColor(gui.black)
    gg.fillRect(
      cellSize.width * intToDouble(state.c),
      cellSize.height * intToDouble(state.r),
      cellSize.width,
      cellSize.height,
    )
  }
}

void main() {
  state.r = 4
  state.c = 4
  state.R = 10
  state.C = 10

  gui.start(gui.Options()
    .setPos(gui.Point(0.0, 0.0))
    .setSize(gui.Size(400.0, 400.0))
    .setResizable(false)
    .onKey(def bool[](gui.KeyEvent e) {
      bool changed = false
      if (e.chars == 'a' and state.c > 0) {
        state.c = state.c - 1
        changed = true
      } else if (e.chars == 'w' and state.r + 1 < state.R) {
        state.r = state.r + 1
        changed = true
      } else if (e.chars == 's' and state.r > 0) {
        state.r = state.r - 1
        changed = true
      } else if (e.chars == 'd' and state.c + 1 < state.C) {
        state.c = state.c + 1
        changed = true
      }
      if (changed) {
        redraw()
      }
      return changed
    })
    .onStart(def void[]() {
      print(gui.monospaceFont)
      final dim = gui.getSize()
      with gui.graphics() as gg {
        print('onStart')
        gg.setFillColor(gui.Color(1.0, 1.0, 0.5, 1.0))
        gg.fillRect(0.0, 0.0, dim.width, dim.height)
        gg.setTextForegroundColor(gui.green)
        gg.setTextBackgroundColor(gui.red)
        gg.setFont(gui.monospaceFont)
        final size = gg.fillTextSize('Hello world!')
        final twidth = size.width
        final theight = size.height
        gg.fillText(dim.width / 2.0, dim.height / 2.0 + theight, 'Hello world!')
        gg.fillText(dim.width / 2.0 + twidth, dim.height / 2.0, 'again')
        gg.fillText(dim.width / 2.0, dim.height / 2.0 - 1.0 * theight, '世')
        gg.fillText(dim.width / 2.0, dim.height / 2.0 - 2.0 * theight, '世界!')
        print('H -> ' + str(gg.fillTextSize('H')))
        print('e -> ' + str(gg.fillTextSize('e')))
        print('世 -> ' + str(gg.fillTextSize('世')))
        print('界 -> ' + str(gg.fillTextSize('界')))
        gg.fillText(
          dim.width / 4.0,
          dim.height / 2.0 + 3.0 * theight,
          'Press WASD to start')
      }
    }))
}
