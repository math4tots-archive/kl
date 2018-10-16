#include "klc_posix.h"

#if KLC_POSIX
#include <unistd.h>
#endif

KLCNPOSIXInterface* KLCN_initPOSIX() {
#if KLC_POSIX
  KLCNPOSIXInterface* i = (KLCNPOSIXInterface*) malloc(sizeof(KLCNPOSIXInterface));
  KLC_init_header(&i->header, &KLC_typePOSIXInterface);
  return i;
#else
  return NULL;
#endif
}

void KLC_deletePOSIXInterface(KLC_header* robj, KLC_header** dq) {
}

KLCNString* KLCNPOSIXInterface_mgetcwd(KLCNPOSIXInterface* p) {
#if KLC_POSIX
  size_t cap = 2, len;
  char* buffer = (char*) malloc(sizeof(char) * cap);
  while (getcwd(buffer, cap) == NULL) {
    if (errno != ERANGE) {
      /* We failed to getcwd for a reason besides not big enough buffer
       * TODO: Better error handling */
      return NULL;
    }
    cap *= 2;
    buffer = (char*) realloc(buffer, sizeof(char) * cap);
  }
  len = strlen(buffer);
  buffer = (char*) realloc(buffer, sizeof(char) * (len + 1));
  return KLC_mkstr_with_buffer(len, buffer, KLC_check_ascii(buffer));
#else
  return NULL;
#endif
}
