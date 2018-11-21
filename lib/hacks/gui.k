// Dead simple single window desktop GUI API
// There are some limitations for the sake of simplicity:
//   1. No explicit multi-threading.
//      There may be multi-threading, but you can't
//      control it.
//   2. Only one window.
//      This is an API design issue.
//      And I'm not sure how annoying this would be to
//      implement on all platforms yet.

#(append C_HEADERS 'klc_hacks_gui.h')

#(if
  (eq PLATFORM 'darwin') (begin
    (append C_SOURCES 'klc_hacks_gui_osx.m')
    (append APPLE_FRAMEWORKS 'Cocoa')
  )
  (error (add 'Unsupported platform: ' PLATFORM))
)

_WithGraphicsContext graphics() {
  return api.graphics()
}

void start(Options opts) {
  api.start(opts)
}

extern Try tryApi

Api api = tryApi.orDie().(Api)

extern class Api {
  // start main loop: this function does not return
  // starts the application
  void start(Options options)

  // These three methods should be used together properly
  // Prefer using the 'graphics()' method rather than
  // calling these methods directly
  GraphicsContext _getGc()
  void _beginDraw()
  void _endDraw()

  _WithGraphicsContext graphics() {
    return _WithGraphicsContext(this)
  }
}

class _WithGraphicsContext {
  Api api
  WeakReference wr
  GraphicsContext gc

  new(Api a) {
    this.api = a
  }

  GraphicsContext Enter() {
    this.api._beginDraw()
    final gc = this.api._getGc()
    this.wr = WeakReference(gc)
    return gc
  }

  void Exit() {
    this.api._endDraw()
    if (this.wr) {
      panic('GraphicsContext reference escaped!')
    }
  }
}

class Options {
  bool resizable
  int x
  int y
  int width
  int height
  var startCallback
  var keyCallback
  var modifierKeyCallback

  new() {
    this.x = this.y = 0
    this.width = this.height = 300
  }

  Options setResizable(bool resizable) {
    this.resizable = resizable
    return this
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

  Options onStart(var callback) {
    this.startCallback = callback
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

case class Rect {
  Point point
  Size size
}

case class Point {
  double x
  double y
}

case class Size {
  double width
  double height
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
  return bool(try.getFont(name, 12))
}

try Font getFont(String name, int size)

Font monospaceFont = (
  fontExists('Courier New') ?
    getFont('Courier New', 12) :
  fontExists('Monaco') ?
    getFont('Monaco', 12) :
    getFont('Courier', 12)
)
