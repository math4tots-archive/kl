
// TODO: Define build configurations for other environments as well
#(append APPLE_FRAMEWORKS 'SDL2')
#(append C_HEADERS 'klc_sdl.h')
#(append C_SOURCES 'klc_sdl.c')

extern class SDLContext {
  int init(int flags)
  void quit()
  void delay(int millis)
  int GETINIT_TIMER()
  int GETINIT_AUDIO()
  int GETINIT_VIDEO()
  int GETINIT_JOYSTICK()
  int GETINIT_HAPTIC()
  int GETINIT_GAMECONTROLLER()
  int GETINIT_EVENTS()
  int GETINIT_EVERYTHING()
  int GETINIT_NOPARACHUTE()
}

extern SDLContext sdl
