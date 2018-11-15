#(append C_HEADERS 'klc_hacks_gui.h')

#(if
  (eq PLATFORM 'win32') (begin
    (append C_SOURCES 'klc_hacks_gui_windows.c')
  )
  (error (add 'Unsupported platform: ' PLATFORM))
)

extern class Api {
  // Dead simple dialog box that will show a message
  // to the user.
  // Synchronous -- blocks until user exits the message box.
  void alert(String message, String title)
}

extern Api api
