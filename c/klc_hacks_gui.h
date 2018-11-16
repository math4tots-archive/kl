#ifndef klc_os_fs_h
#define klc_os_fs_h

#include "klc_prelude.h"

typedef struct KLCNhacksZBguiZBApi KLCNhacksZBguiZBApi;
typedef struct KLCNhacksZBguiZBWindow KLCNhacksZBguiZBWindow;

extern KLC_typeinfo KLC_typehacksZBguiZBApi;
extern KLC_typeinfo KLC_typehacksZBguiZBWindow;

void KLC_deletehacksZBguiZBApi(KLC_header*, KLC_header**);
void KLC_deletehacksZBguiZBWindow(KLC_header*, KLC_header**);

KLCNhacksZBguiZBApi* KLCNhacksZBguiZBapiZEinit();

void KLCNhacksZBguiZBApiZFalert(KLCNhacksZBguiZBApi* api, KLCNString* message);
void KLCNhacksZBguiZBApiZFmain(KLCNhacksZBguiZBApi* api);
KLCNTry* KLCNhacksZBguiZBApiZFwindowZDtry(
  KLCNhacksZBguiZBApi* api, KLCNString* title, KLC_int width, KLC_int height);

void KLCNhacksZBguiZBWindowZFshow(KLCNhacksZBguiZBWindow* win);
void KLCNhacksZBguiZBWindowZFupdate(KLCNhacksZBguiZBWindow* win);

#endif/*klc_os_fs_h*/
