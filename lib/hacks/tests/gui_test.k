import hacks.gui as GUI


void main() {
  final gui = GUI.api
  // gui.alert('Hello 世界!')
  // gui.alert('This is some message to show')
  gui.start(GUI.Options()
    .setPos(0, 0)
    .setSize(300, 400)
    .onDraw(def void[](GUI.GraphicsContext gc) {
      print('gc = ' + str(gc))
      gc.setFillColor(GUI.blue)
      gc.fillBackground()
      gc.setFillColor(GUI.black)
      gc.fillRect(0.0, 0.0, gc.width / 2.0, gc.height / 2.0)
    })
  )
}
