#(append C_HEADERS 'klc_hacks_gui.h')

#(if
  (eq PLATFORM 'win32') (begin
    (append C_SOURCES 'klc_hacks_gui_windows.c')
  )
  (eq PLATFORM 'darwin') (begin
    (append C_SOURCES 'klc_hacks_gui_osx.m')
    (append APPLE_FRAMEWORKS 'Cocoa')
  )
  (error (add 'Unsupported platform: ' PLATFORM))
)

extern Api api

extern class Api {
  // Dead simple dialog box that will show a message
  // to the user.
  // Synchronous -- blocks until user exits the message box.
  // Mostly for debugging
  void alert(String message)

  // main loop: this function does not return
  void main()

  // Creates a new window
  try Window window(String title, int width, int height)
}

extern class Window {
  void show()
  void update()
}
