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

  Options onDraw(var drawCallback) {
    this.drawCallback = drawCallback
    return this
  }
}

extern class GraphicsContext {
  double GETwidth()
  double GETheight()
  void setFillColor(Color color)
  void fillRect(double x, double y, double width, double height)
  void fillBackground() {
    this.fillRect(0.0, 0.0, this.width, this.height)
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

