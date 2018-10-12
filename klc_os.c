#include "klc_os.h"

KLCNOperatingSystemInterface* KLCN_initos() {
  KLCNOperatingSystemInterface* os =
    (KLCNOperatingSystemInterface*) malloc(sizeof(KLCNOperatingSystemInterface));
  KLC_init_header(&os->header, &KLC_typeOperatingSystemInterface);
  return os;
}

void KLC_deleteOperatingSystemInterface(KLC_header* robj, KLC_header** dq) {
}

KLCNString* KLCNOperatingSystemInterface_mGETname(KLCNOperatingSystemInterface* os) {
  return KLC_mkstr(KLC_OS_NAME);
}

KLC_bool KLCNOperatingSystemInterface_mGETposix(KLCNOperatingSystemInterface* os) {
#if KLC_POSIX
  return 1;
#else
  return 0;
#endif
}

