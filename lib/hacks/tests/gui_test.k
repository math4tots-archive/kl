import hacks.gui as GUI


void main() {
  final gui = GUI.api
  gui.alert('Hello 世界!')
  gui.alert('This is some message to show')
  final window = gui.window('XXX', 400, 300)
  print(window)
  window.show()
  window.update()
  gui.main()
}
