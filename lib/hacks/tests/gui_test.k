import hacks.gui as GUI


void main() {
  final gui = GUI.api
  // gui.alert('Hello 世界!')
  // gui.alert('This is some message to show')
  // print(GUI.try.getFont('monaco', 12))
  print(GUI.getFont('monaco', 12))
  gui.start(GUI.Options()
    .setPos(0, 0)
    .setSize(300, 400)
    .onKey(def void[GUI.Api gui](GUI.KeyEvent e) {
      with gui.graphics() as gc {
        print(e.chars)
        if (e.chars == 'r') {
          print('color red')
          gc.setFillColor(GUI.red)
        } else if (e.chars == 'b') {
          print('color blue')
          gc.setFillColor(GUI.blue)
        } else {
          print('color green')
          gc.setFillColor(GUI.green)
        }
        double woff = e.chars == 'g' ? 100.0 : 0.0
        gc.fillRect(0.0, 0.0, 300.0 / 2.0 + woff, 400.0 / 2.0)
      }
    })
    .onStart(def void[](GUI.GraphicsContext gc) {
      print('onStart')
      gc.setFillColor(GUI.Color(1.0, 1.0, 0.5, 1.0))
      gc.fillRect(0.0, 0.0, gc.width, gc.height)
      gc.setTextForegroundColor(GUI.green)
      gc.setTextBackgroundColor(GUI.red)
      gc.setFont(GUI.monospaceFont)
      final size = gc.fillTextSize('Hello world!')
      final twidth = size.width
      final theight = size.height
      gc.fillText(gc.width / 2.0, gc.height / 2.0 + theight, 'Hello world!')
      gc.fillText(gc.width / 2.0 + twidth, gc.height / 2.0, 'again')
      gc.fillText(gc.width / 2.0, gc.height / 2.0 - 1.0 * theight, '世')
      gc.fillText(gc.width / 2.0, gc.height / 2.0 - 2.0 * theight, '世界!')
      print('H -> ' + str(gc.fillTextSize('H')))
      print('e -> ' + str(gc.fillTextSize('e')))
      print('世 -> ' + str(gc.fillTextSize('世')))
      print('界 -> ' + str(gc.fillTextSize('界')))
    })
  )
}
