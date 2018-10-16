#ifndef klc_cocoa_h
#define klc_cocoa_h

#include "klc_prelude.h"

typedef struct KLCNCOCOAInterface KLCNCOCOAInterface;

struct KLCNCOCOAInterface {
  KLC_header header;
};

KLCNCOCOAInterface* KLCN_initCOCOA();
void KLC_deleteCOCOAInterface(KLC_header*, KLC_header**);

#endif/*klc_cocoa_h*/
