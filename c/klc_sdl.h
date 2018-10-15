#ifndef klc_sdl_h
#define klc_sdl_h
#include "klc_prelude.h"

typedef struct KLCNSDLContext KLCNSDLContext;

struct KLCNSDLContext {
  KLC_header header;
};

KLC_int KLCNSDLContext_minit(KLCNSDLContext*, KLC_int KLCNflags);
KLCNSDLContext* KLCN_initsdl();
void KLC_deleteSDLContext(KLC_header*, KLC_header**);
void KLCNSDLContext_mquit(KLCNSDLContext*);
void KLCNSDLContext_mdelay(KLCNSDLContext*, KLC_int);
KLC_int KLCNSDLContext_mGETINIT_TIMER(KLCNSDLContext*);
KLC_int KLCNSDLContext_mGETINIT_AUDIO(KLCNSDLContext*);
KLC_int KLCNSDLContext_mGETINIT_VIDEO(KLCNSDLContext*);
KLC_int KLCNSDLContext_mGETINIT_JOYSTICK(KLCNSDLContext*);
KLC_int KLCNSDLContext_mGETINIT_HAPTIC(KLCNSDLContext*);
KLC_int KLCNSDLContext_mGETINIT_GAMECONTROLLER(KLCNSDLContext*);
KLC_int KLCNSDLContext_mGETINIT_EVENTS(KLCNSDLContext*);
KLC_int KLCNSDLContext_mGETINIT_EVERYTHING(KLCNSDLContext*);
KLC_int KLCNSDLContext_mGETINIT_NOPARACHUTE(KLCNSDLContext*);
#endif/*klc_sdl_h*/
