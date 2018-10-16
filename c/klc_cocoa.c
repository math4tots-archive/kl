#include "klc_cocoa.h"
/* dummy c file for platforms that don't have cocoa */

KLCNCOCOAInterface* KLCN_initCOCOA() {
  return NULL;
}

void KLC_deleteCOCOAInterface(KLC_header*, KLC_header**) {
}
