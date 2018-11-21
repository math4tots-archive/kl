import hacks.gui


void main() {
  gui.start(gui.Options()
    .setPos(0, 0)
    .setSize(300, 400)
    .setResizable(true)
    .onKey(def void[](gui.KeyEvent e) {
      with gui.graphics() as gg {
        print(e.chars)
        if (e.chars == 'r') {
          print('color red')
          gg.setFillColor(gui.red)
        } else if (e.chars == 'b') {
          print('color blue')
          gg.setFillColor(gui.blue)
        } else {
          print('color green')
          gg.setFillColor(gui.green)
        }
        double woff = e.chars == 'g' ? 100.0 : 0.0
        gg.fillRect(0.0, 0.0, 300.0 / 2.0 + woff, 400.0 / 2.0)
      }
    })
    .onStart(def void[]() {
      print(gui.monospaceFont)
      with gui.graphics() as gg {
        print('onStart')
        gg.setFillColor(gui.Color(1.0, 1.0, 0.5, 1.0))
        gg.fillRect(0.0, 0.0, gg.width, gg.height)
        gg.setTextForegroundColor(gui.green)
        gg.setTextBackgroundColor(gui.red)
        gg.setFont(gui.monospaceFont)
        final size = gg.fillTextSize('Hello world!')
        final twidth = size.width
        final theight = size.height
        gg.fillText(gg.width / 2.0, gg.height / 2.0 + theight, 'Hello world!')
        gg.fillText(gg.width / 2.0 + twidth, gg.height / 2.0, 'again')
        gg.fillText(gg.width / 2.0, gg.height / 2.0 - 1.0 * theight, '世')
        gg.fillText(gg.width / 2.0, gg.height / 2.0 - 2.0 * theight, '世界!')
        print('H -> ' + str(gg.fillTextSize('H')))
        print('e -> ' + str(gg.fillTextSize('e')))
        print('世 -> ' + str(gg.fillTextSize('世')))
        print('界 -> ' + str(gg.fillTextSize('界')))
      }
    }))
}
