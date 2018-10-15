#include "./sdl.k"

void main() {
  if (sdl.init(sdl.INIT_VIDEO) != 0) {
    panic('SDL_Init error')
  }
  sdl.delay(3000)
  sdl.quit()
}
