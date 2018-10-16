#ifndef klc_posix_h
#define klc_posix_h

#include "klc_prelude.h"

typedef struct KLCNPOSIXInterface KLCNPOSIXInterface;

extern KLC_typeinfo KLC_typePOSIXInterface;

struct KLCNPOSIXInterface {
  KLC_header header;
};

KLCNPOSIXInterface* KLCN_initPOSIX();
void KLC_deletePOSIXInterface(KLC_header*, KLC_header**);
KLC_int KLCNPOSIXInterface_mGETVERSION(KLCNPOSIXInterface*);
KLCNString* KLCNPOSIXInterface_mgetcwd(KLCNPOSIXInterface*);
KLCNList* KLCNPOSIXInterface_mreaddir(KLCNPOSIXInterface*, KLCNString*);

#endif/*klc_posix_h*/
