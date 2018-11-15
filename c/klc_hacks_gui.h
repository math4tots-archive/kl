#ifndef klc_os_fs_h
#define klc_os_fs_h

#include "klc_prelude.h"

typedef struct KLCNhacksZBguiZBApi KLCNhacksZBguiZBApi;

extern KLC_typeinfo KLC_typehacksZBguiZBApi;

void KLC_deletehacksZBguiZBApi(KLC_header*, KLC_header**);
KLCNhacksZBguiZBApi* KLCNhacksZBguiZBapiZEinit();

void KLCNhacksZBguiZBApiZFalert(KLCNhacksZBguiZBApi* api, KLCNString* message);

void KLCNhacksZBguiZBApiZFmkwin(KLCNhacksZBguiZBApi* api);

#endif/*klc_os_fs_h*/
