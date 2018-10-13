#ifndef klc_os_h
#define klc_os_h
#include "klc_prelude.h"


typedef struct KLCNOperatingSystemInterface KLCNOperatingSystemInterface;

struct KLCNOperatingSystemInterface {
  KLC_header header;
};

extern KLC_typeinfo KLC_typeOperatingSystemInterface;

KLCNOperatingSystemInterface* KLCN_initos();
void KLC_deleteOperatingSystemInterface(KLC_header* robj, KLC_header** dq);
KLCNString* KLCNOperatingSystemInterface_mGETname(KLCNOperatingSystemInterface*);
KLC_bool KLCNOperatingSystemInterface_mGETposix(KLCNOperatingSystemInterface*);
KLC_bool KLCNOperatingSystemInterface_mBool(KLCNOperatingSystemInterface*);
KLCNList* KLCNOperatingSystemInterface_mlistdir(
  KLCNOperatingSystemInterface*,
  KLCNString*);
KLC_bool KLCNOperatingSystemInterface_misfile(
    KLCNOperatingSystemInterface* os,
    KLCNString* path);
KLC_bool KLCNOperatingSystemInterface_misdir(
    KLCNOperatingSystemInterface* os,
    KLCNString* path);

#endif/*klc_os_h*/
