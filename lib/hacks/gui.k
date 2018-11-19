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

  Options setDrawCallback(var drawCallback) {
    this.drawCallback = drawCallback
    return this
  }
}
