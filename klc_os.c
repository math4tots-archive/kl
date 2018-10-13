#include "klc_os.h"

#include <string.h>

#if KLC_POSIX
#include <unistd.h>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#elif KLC_OS_WINDOWS
#include <windows.h>
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
#elif KLC_OS_WINDOWS
  /* TODO: Use unicode versions of these functions */
  KLCNString* suffix = KLC_mkstr("\\*");
  KLCNString* pattern = KLCNString_mAdd(path, suffix);
  WIN32_FIND_DATAA data;
  HANDLE hFind = FindFirstFileA(pattern->buffer, &data);
  KLCNList* ret = KLC_mklist(0);
  if (hFind == INVALID_HANDLE_VALUE) {
    /* Read failed. TODO: Better error handling */
    return NULL;
  }
  do {
    KLC_header* name = (KLC_header*) KLC_mkstr(data.cFileName);
    KLCNList_mpush(ret, KLC_object_to_var(name));
    KLC_release(name);
  } while (FindNextFileA(hFind, &data));
  FindClose(hFind);
  KLC_release((KLC_header*) suffix);
  KLC_release((KLC_header*) pattern);
  return ret;
#else
  KLC_errorf("os.listdir not supported for %s", KLC_OS_NAME);
  return NULL;
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
#elif KLC_OS_WINDOWS
  DWORD bufsize;
  LPWSTR buf;
  bufsize = GetCurrentDirectoryW(0, NULL);
  buf = (LPWSTR) malloc(bufsize);
  GetCurrentDirectoryW(bufsize, buf);
  return KLC_windows_string_from_wstr_buffer(buf, bufsize);
#else
  KLC_errorf("os.getcwd not supported for %s", KLC_OS_NAME);
  return NULL;
#endif
}

KLC_bool KLCNOperatingSystemInterface_misfile(
    KLCNOperatingSystemInterface* os,
    KLCNString* path) {
#if KLC_POSIX
  struct stat sb;
  if (stat(path->buffer, &sb) == 0) {
    /* Returns true iff we check that this path is a regular file */
    return !!S_ISREG(sb.st_mode);
  }
  /* We can't prove that given path is a file, so we return false. */
  return 0;
#elif KLC_OS_WINDOWS
  /* TODO: Use the unicode version of this function */
  DWORD ftype = GetFileAttributesA(path->buffer);
  if (ftype == INVALID_FILE_ATTRIBUTES) {
    return 0;
  }
  if (!(ftype & FILE_ATTRIBUTE_DIRECTORY)) {
    return 1;
  }
  return 0;
#else
  /* TODO: The best we can do in an unknown environment,
   * is just to open the file and see if it succeeds */
  KLC_errorf("os.isfile not supported for %s", KLC_OS_NAME);
  return 0;
#endif
}

KLC_bool KLCNOperatingSystemInterface_misdir(
    KLCNOperatingSystemInterface* os,
    KLCNString* path) {
#if KLC_POSIX
  struct stat sb;
  if (stat(path->buffer, &sb) == 0) {
    /* Returns true iff we check that this path is a regular file */
    return !!S_ISDIR(sb.st_mode);
  }
  /* We can't prove that given path is a file, so we return false. */
  return 0;
#elif KLC_OS_WINDOWS
  /* TODO: Use the unicode version of this function */
  DWORD ftype = GetFileAttributesA(path->buffer);
  if (ftype == INVALID_FILE_ATTRIBUTES) {
    return 0;
  }
  if (ftype & FILE_ATTRIBUTE_DIRECTORY) {
    return 1;
  }
  return 0;
#else
  /* TODO: The best we can do in an unknown environment,
   * is just to open the file and see if it succeeds */
  KLC_errorf("os.isdir not supported for %s", KLC_OS_NAME);
  return 0;
#endif
}
