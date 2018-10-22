#ifndef klc_os_h
#define klc_os_h
#include "klc_prelude.h"


void KLC_deleteosZBInterface(KLC_header* robj, KLC_header** dq);
KLCNString* KLCNosZBpathZBnameZEinit();
KLCNString* KLCNosZBpathZBsepZEinit();
KLCNList* KLCNosZBlistdir(KLCNString*);
KLC_bool KLCNosZBisfile(KLCNString* path);
KLC_bool KLCNosZBisdir(KLCNString* path);

#endif/*klc_os_h*/
