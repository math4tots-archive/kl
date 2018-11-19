// Dead simple single window desktop GUI API

#(append C_HEADERS 'klc_hacks_gui.h')

#(if
  (eq PLATFORM 'darwin') (begin
    (append C_SOURCES 'klc_hacks_gui_osx.m')
    (append APPLE_FRAMEWORKS 'Cocoa')
  )
  (error (add 'Unsupported platform: ' PLATFORM))
)

extern Try tryApi

Api api = tryApi.orDie().(Api)

extern class Api {
  // Dead simple dialog box that will show a message
  // to the user.
  // Synchronous -- blocks until user exits the message box.
  // Mostly for debugging
  void alert(String message)

  // start main loop: this function does not return
  // starts the application
  void start(Options options)
}

class Options {
  int x
  int y
  int width
  int height
  var drawCallback
  var keyCallback
  var modifierKeyCallback

  new() {
    this.x = this.y = 0
    this.width = this.height = 300
  }

  Options setPos(int x, int y) {
    this.x = x
    this.y = y
    return this
  }

  Options setSize(int width, int height) {
    this.width = width
    this.height = height
    return this
  }

  Options onDraw(var callback) {
    this.drawCallback = callback
    return this
  }

  Options onKey(var callback) {
    this.keyCallback = callback
    return this
  }

  Options onModifierKey(var callback) {
    this.modifierKeyCallback = callback
    return this
  }
}

extern class GraphicsContext {
  double GETwidth()
  double GETheight()
  List fillTextSizeAsList(String text)  // [width, height] of bounding rect
  void setFillColor(Color color)
  void setFont(Font font)
  void setTextForegroundColor(Color color)
  void setTextBackgroundColor(Color color)
  void fillRect(double x, double y, double width, double height)
  void fillText(double x, double y, String text)

  void fillBackground() {
    this.fillRect(0.0, 0.0, this.width, this.height)
  }

  Size fillTextSize(String text) {
    final dim = this.fillTextSizeAsList(text)
    return Size(dim[0].(double), dim[1].(double))
  }
}

class Size {
  double width
  double height
  new(double width, double height) {
    this.width = width
    this.height = height
  }

  String Repr() {
    return 'Size(' + str(this.width) + ', ' + str(this.height) + ')'
  }
}

List keyModifiers = [
  'shift',
  'control',
  'alt',
  'command',
]

extern class KeyEvent {
  // Characters associated with the key event
  String GETchars()
  bool GetItem(String modifier)

  String Repr() {
    final liveMods = []
    for String mod in keyModifiers {
      if (this[mod]) {
        liveMods.push(mod)
      }
    }
    final modifiers = '|'.join(liveMods)
    if (modifiers) {
      return 'KeyEvent(' + repr(this.chars) + ', ' + modifiers + ')'
    } else {
      return 'KeyEvent(' + repr(this.chars) + ')'
    }
  }
}

// TODO: Make this an extern class so that platform specific
// color objects can be cached here.
class Color {
  double r
  double g
  double b
  double a

  new(double r, double g, double b, double a) {
    this.r = r
    this.g = g
    this.b = b
    this.a = a
  }

  String Repr() {
    return 'Color(' + ', '.join([
      this.r, this.g, this.b, this.a,
    ].map(repr)) + ')'
  }
}

Color black = Color(0.0, 0.0, 0.0, 1.0)
Color white = Color(1.0, 1.0, 1.0, 1.0)
Color red = Color(1.0, 0.0, 0.0, 1.0)
Color green = Color(0.0, 1.0, 0.0, 1.0)
Color blue = Color(0.0, 0.0, 1.0, 1.0)

extern class Font {
  String GETname()
  int GETsize()
  String Repr() {
    return 'Font(' + repr(this.name) + ', ' + str(this.size) + ')'
  }
}

bool fontExists(String name) {
  return bool(getFont(name, 12))
}

try Font getFont(String name, int size)

Font monospaceFont = (
  fontExists('Monaco') ?
    getFont('Monaco', 12) :
    getFont('Courier', 12)
)
