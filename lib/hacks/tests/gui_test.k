import hacks.gui as GUI


void main() {
  final gui = GUI.api
  // gui.alert('Hello 世界!')
  // gui.alert('This is some message to show')
  gui.start(GUI.Options()
    .setPos(0, 0)
    .setSize(300, 400))
}
