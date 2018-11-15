#include "klc_hacks_gui.h"
#include <windows.h>

#pragma comment(lib, "User32.lib")

struct KLCNhacksZBguiZBApi {
  KLC_header header;
  HINSTANCE hInstance;
};

void KLC_deletehacksZBguiZBApi(KLC_header* win, KLC_header** dq) {
}

KLCNhacksZBguiZBApi* KLCNhacksZBguiZBapiZEinit() {
  KLCNhacksZBguiZBApi* win =
    (KLCNhacksZBguiZBApi*) malloc(sizeof(KLCNhacksZBguiZBApi));
  KLC_init_header(&win->header, &KLC_typehacksZBguiZBApi);
  win->hInstance = GetModuleHandle(NULL);
  return win;
}

void KLCNhacksZBguiZBApiZFalert(KLCNhacksZBguiZBApi* api, KLCNString* message) {
  MessageBoxW(
    NULL,
    KLC_windows_get_wstr(message),
    L"Alert",
    0);
}
