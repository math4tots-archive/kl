#ifndef klc_os_h
#define klc_os_h
#include "klc_prelude.h"


typedef struct KLCNosZBInterface KLCNosZBInterface;

struct KLCNosZBInterface {
  KLC_header header;
};

extern KLC_typeinfo KLC_typeosZBInterface;

KLCNosZBInterface* KLCNOSZEinit();
void KLC_deleteosZBInterface(KLC_header* robj, KLC_header** dq);
KLCNString* KLCNosZBInterfaceZFGETname(KLCNosZBInterface*);
KLC_bool KLCNosZBInterfaceZFGETposix(KLCNosZBInterface*);
KLC_bool KLCNosZBInterfaceZFBool(KLCNosZBInterface*);
KLCNList* KLCNosZBInterfaceZFlistdir(
  KLCNosZBInterface*,
  KLCNString*);
KLC_bool KLCNosZBInterfaceZFisfile(
    KLCNosZBInterface* os,
    KLCNString* path);
KLC_bool KLCNosZBInterfaceZFisdir(
    KLCNosZBInterface* os,
    KLCNString* path);

#endif/*klc_os_h*/
