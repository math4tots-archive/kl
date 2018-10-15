#include "klc_sdl.h"
#include <SDL2/SDL.h>

extern KLC_typeinfo KLC_typeSDLContext;

KLCNSDLContext* KLCN_initsdl() {
  KLCNSDLContext* sdl = (KLCNSDLContext*) malloc(sizeof(KLCNSDLContext));
  KLC_init_header(&sdl->header, &KLC_typeSDLContext);
  return sdl;
}

KLC_int KLCNSDLContext_minit(KLCNSDLContext* sdl, KLC_int flags) {
  return (KLC_int) SDL_Init(flags);
}

void KLCNSDLContext_mquit(KLCNSDLContext* sdl) {
  SDL_Quit();
}

void KLCNSDLContext_mdelay(KLCNSDLContext* sdl, KLC_int millis) {
  SDL_Delay(millis);
}

KLC_int KLCNSDLContext_mGETINIT_TIMER(KLCNSDLContext* sdl) {
  return (KLC_int) SDL_INIT_TIMER;
}

KLC_int KLCNSDLContext_mGETINIT_AUDIO(KLCNSDLContext* sdl) {
  return (KLC_int) SDL_INIT_AUDIO;
}

KLC_int KLCNSDLContext_mGETINIT_VIDEO(KLCNSDLContext* sdl) {
  return (KLC_int) SDL_INIT_VIDEO;
}

KLC_int KLCNSDLContext_mGETINIT_JOYSTICK(KLCNSDLContext* sdl) {
  return (KLC_int) SDL_INIT_JOYSTICK;
}

KLC_int KLCNSDLContext_mGETINIT_HAPTIC(KLCNSDLContext* sdl) {
  return (KLC_int) SDL_INIT_HAPTIC;
}

KLC_int KLCNSDLContext_mGETINIT_GAMECONTROLLER(KLCNSDLContext* sdl) {
  return (KLC_int) SDL_INIT_GAMECONTROLLER;
}

KLC_int KLCNSDLContext_mGETINIT_EVENTS(KLCNSDLContext* sdl) {
  return (KLC_int) SDL_INIT_EVENTS;
}

KLC_int KLCNSDLContext_mGETINIT_EVERYTHING(KLCNSDLContext* sdl) {
  return (KLC_int) SDL_INIT_EVERYTHING;
}

KLC_int KLCNSDLContext_mGETINIT_NOPARACHUTE(KLCNSDLContext* sdl) {
  return (KLC_int) SDL_INIT_NOPARACHUTE;
}


void KLC_deleteSDLContext(KLC_header* sdl, KLC_header** dq) {
}
