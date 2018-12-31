#ifndef klc_os_fs_h
#define klc_os_fs_h

#include "klc_prelude.h"

KLCNString* KLCNosZBfsZBsepZEinit(void);
KLC_bool KLCNosZBfsZBisdir(KLCNString* path);
KLC_bool KLCNosZBfsZBisfile(KLCNString* path);
KLCNTry* KLCNosZBfsZBtryChdir(KLCNString* path);
KLCNTry* KLCNosZBfsZBtryMkdir(KLCNString* path);
KLCNTry* KLCNosZBfsZBtryListdir(KLCNString* path);
KLCNTry* KLCNosZBfsZBtryGetcwd();

#endif/*klc_os_fs_h*/
