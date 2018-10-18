#include "klc_posix.h"

#if KLC_POSIX
#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
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

KLC_int KLCNPOSIXInterface_mGETVERSION(KLCNPOSIXInterface* p) {
#if KLC_POSIX
  return _POSIX_VERSION;
#else
  return 0;
#endif
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

KLCNList* KLCNPOSIXInterface_mreaddir(KLCNPOSIXInterface* p, KLCNString* path) {
#if KLC_POSIX
  DIR* d;
  struct dirent* dir;
  KLCNList* ret = KLC_mklist(0);
  d = opendir(path->utf8);
  if (!d) {
    /* Read failed. TODO: Better error handling */
    return NULL;
  }
  while ((dir = readdir(d)) != NULL) {
    KLC_header* dirname = (KLC_header*) KLC_mkstr(dir->d_name);
    ino_t ino = dir->d_ino;
    KLCNList* pair = KLC_mklist(2);
    KLCNList_mpush(pair, KLC_int_to_var((KLC_int) ino));
    KLCNList_mpush(pair, KLC_object_to_var(dirname));
    KLCNList_mpush(ret, KLC_object_to_var((KLC_header*) pair));
    KLC_release(dirname);
    KLC_release((KLC_header*) pair);
  }
  closedir(d);
  return ret;
#else
  return NULL;
#endif
}
