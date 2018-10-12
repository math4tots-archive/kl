#include "klc_os.h"

#include <string.h>

#if KLC_POSIX
#include <unistd.h>
#include <dirent.h>
#include <errno.h>
#endif

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

KLC_bool KLCNOperatingSystemInterface_mBool(KLCNOperatingSystemInterface* os) {
#if KLC_OS_UNKNOWN
  return 0;
#else
  return 1;
#endif
}

KLCNList* KLCNOperatingSystemInterface_mlistdir(
    KLCNOperatingSystemInterface* os,
    KLCNString* path) {
#if KLC_POSIX
  DIR* d;
  struct dirent* dir;
  KLCNList* ret = KLC_mklist(0);
  d = opendir(path->buffer);
  if (!d) {
    /* Read failed. TODO: Better error handling */
    return NULL;
  }
  while ((dir = readdir(d)) != NULL) {
    KLC_header* dirname = (KLC_header*) KLC_mkstr(dir->d_name);
    KLCNList_mpush(ret, KLC_object_to_var(dirname));
    KLC_release(dirname);
  }
  closedir(d);
  return ret;
#else
  KLC_errorf("os.listdir not supported for %s", KLC_OS_NAME);
#endif
}

KLCNString* KLCNOperatingSystemInterface_mgetcwd(KLCNOperatingSystemInterface* os) {
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
  KLC_errorf("os.getcwd not supported for %s", KLC_OS_NAME);
#endif
}

