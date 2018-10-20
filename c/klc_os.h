#ifndef klc_os_h
#define klc_os_h
#include "klc_prelude.h"


typedef struct KLCNOperatingSystemInterface KLCNOperatingSystemInterface;

struct KLCNOperatingSystemInterface {
  KLC_header header;
};

extern KLC_typeinfo KLC_typeOperatingSystemInterface;

KLCNOperatingSystemInterface* KLCNOSZEinit();
void KLC_deleteOperatingSystemInterface(KLC_header* robj, KLC_header** dq);
KLCNString* KLCNOperatingSystemInterfaceZFGETname(KLCNOperatingSystemInterface*);
KLC_bool KLCNOperatingSystemInterfaceZFGETposix(KLCNOperatingSystemInterface*);
KLC_bool KLCNOperatingSystemInterfaceZFBool(KLCNOperatingSystemInterface*);
KLCNList* KLCNOperatingSystemInterfaceZFlistdir(
  KLCNOperatingSystemInterface*,
  KLCNString*);
KLC_bool KLCNOperatingSystemInterfaceZFisfile(
    KLCNOperatingSystemInterface* os,
    KLCNString* path);
KLC_bool KLCNOperatingSystemInterfaceZFisdir(
    KLCNOperatingSystemInterface* os,
    KLCNString* path);

#endif/*klc_os_h*/
