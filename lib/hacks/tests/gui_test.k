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
    .onKey(def void[](GUI.KeyEvent e) {
      print('e = ' + str(e))
    })
    .onDraw(def void[](GUI.GraphicsContext gc) {
      gc.setFillColor(GUI.blue)
      gc.fillBackground()
      gc.setFillColor(GUI.black)
      gc.fillRect(0.0, 0.0, gc.width / 2.0, gc.height / 2.0)
      gc.setTextForegroundColor(GUI.green)
      gc.setFont(GUI.monospaceFont)
      gc.fillText(gc.width / 2.0, gc.height / 2.0, 'Hello world!')

      final size = gc.fillTextSize('Hello world!')
      final twidth = size.width
      final theight = size.height

      gc.setTextForegroundColor(GUI.green)
      gc.setTextBackgroundColor(GUI.red)
      gc.setFont(GUI.monospaceFont)
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
