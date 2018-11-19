#ifndef klc_os_fs_h
#define klc_os_fs_h

#include "klc_prelude.h"

typedef struct KLCNhacksZBguiZBApi KLCNhacksZBguiZBApi;
struct KLCNhacksZBguiZBOptions;

extern KLC_typeinfo KLC_typehacksZBguiZBApi;

void KLC_deletehacksZBguiZBApi(KLC_header*, KLC_header**);

KLCNTry* KLCNhacksZBguiZBtryApiZEinit();

void KLCNhacksZBguiZBApiZFalert(KLCNhacksZBguiZBApi* api, KLCNString* message);
void KLCNhacksZBguiZBApiZFstart(KLCNhacksZBguiZBApi* api, struct KLCNhacksZBguiZBOptions*);

/* BEGIN Function declarations for extracting data from Options */
KLC_int KLCNhacksZBguiZBOptionsZFGETx(struct KLCNhacksZBguiZBOptions* opts);
KLC_int KLCNhacksZBguiZBOptionsZFGETy(struct KLCNhacksZBguiZBOptions* opts);
KLC_int KLCNhacksZBguiZBOptionsZFGETwidth(struct KLCNhacksZBguiZBOptions* opts);
KLC_int KLCNhacksZBguiZBOptionsZFGETheight(struct KLCNhacksZBguiZBOptions* opts);
/* END */

#endif/*klc_os_fs_h*/
