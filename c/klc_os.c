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

KLCNosZBInterface* KLCNosZBOSZEinit() {
  KLCNosZBInterface* os =
    (KLCNosZBInterface*) malloc(sizeof(KLCNosZBInterface));
  KLC_init_header(&os->header, &KLC_typeosZBInterface);
  return os;
}

void KLC_deleteosZBInterface(KLC_header* robj, KLC_header** dq) {
}

KLCNString* KLCNosZBInterfaceZFGETname(KLCNosZBInterface* os) {
  return KLC_mkstr(KLC_OS_NAME);
}

KLC_bool KLCNosZBInterfaceZFGETposix(KLCNosZBInterface* os) {
#if KLC_POSIX
  return 1;
#else
  return 0;
#endif
}

KLCNString* KLCNosZBInterfaceZFGETsep(KLCNosZBInterface* os) {
  return KLC_mkstr(
    #if KLC_OS_WINDOWS
      "\\"
    #else
      "/"
    #endif
  );
}

KLC_bool KLCNosZBInterfaceZFBool(KLCNosZBInterface* os) {
#if KLC_OS_UNKNOWN
  return 0;
#else
  return 1;
#endif
}

KLC_bool KLCNosZBInterfaceZFchdirOrFalse(
    KLCNosZBInterface* os,
    KLCNString* path) {
  #if KLC_POSIX
    return chdir(path->utf8) == 0 ? 1 : 0;
  #elif KLC_OS_WINDOWS
    return SetCurrentDirectoryW(KLC_windows_get_wstr(path)) ? 1 : 0;
  #else
    KLC_errorf("os.chdir not supported for %s", KLC_OS_NAME);
    return NULL;
  #endif
}

KLC_bool KLCNosZBInterfaceZFmkdirOrFalse(
    KLCNosZBInterface* os,
    KLCNString* path) {
  #if KLC_POSIX
    return mkdir(path->utf8, 0777) == 0 ? 1 : 0;
  #elif KLC_OS_WINDOWS
    return CreateDirectoryW(KLC_windows_get_wstr(path), NULL) ? 1 : 0;
  #else
    KLC_errorf("os.mkdir not supported for %s", KLC_OS_NAME);
    return NULL;
  #endif
}

KLCNList* KLCNosZBInterfaceZFlistdirOrNull(
    KLCNosZBInterface* os,
    KLCNString* path) {
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
    KLCNListZFpush(ret, KLC_object_to_var(dirname));
    KLC_release(dirname);
  }
  closedir(d);
  return ret;
#elif KLC_OS_WINDOWS
  /* TODO: Use unicode versions of these functions */
  KLCNString* suffix = KLC_mkstr("\\*");
  KLCNString* pattern = KLCNStringZFAdd(path, suffix);
  WIN32_FIND_DATAW data;
  HANDLE hFind = FindFirstFileW(KLC_windows_get_wstr(pattern), &data);
  KLCNList* ret = KLC_mklist(0);
  if (hFind == INVALID_HANDLE_VALUE) {
    /* Read failed. TODO: Better error handling */
    return NULL;
  }
  do {
    KLC_header* name = (KLC_header*) KLC_windows_string_from_wstr(data.cFileName);
    KLCNListZFpush(ret, KLC_object_to_var(name));
    KLC_release(name);
  } while (FindNextFileW(hFind, &data));
  FindClose(hFind);
  KLC_release((KLC_header*) suffix);
  KLC_release((KLC_header*) pattern);
  return ret;
#else
  KLC_errorf("os.listdir not supported for %s", KLC_OS_NAME);
  return NULL;
#endif
}

KLCNString* KLCNosZBInterfaceZFgetcwdOrNull(KLCNosZBInterface* os) {
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
  bufsize = GetCurrentDirectoryW(0, NULL) + 1;
  buf = (LPWSTR) malloc(bufsize * 2);
  GetCurrentDirectoryW(bufsize, buf);
  return KLC_windows_string_from_wstr_buffer(buf);
#else
  KLC_errorf("os.getcwd not supported for %s", KLC_OS_NAME);
  return NULL;
#endif
}

KLC_bool KLCNosZBInterfaceZFisfile(
    KLCNosZBInterface* os,
    KLCNString* path) {
#if KLC_POSIX
  struct stat sb;
  if (stat(path->utf8, &sb) == 0) {
    /* Returns true iff we check that this path is a regular file */
    return !!S_ISREG(sb.st_mode);
  }
  /* We can't prove that given path is a file, so we return false. */
  return 0;
#elif KLC_OS_WINDOWS
  DWORD ftype = GetFileAttributesW(KLC_windows_get_wstr(path));
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

KLC_bool KLCNosZBInterfaceZFisdir(
    KLCNosZBInterface* os,
    KLCNString* path) {
#if KLC_POSIX
  struct stat sb;
  if (stat(path->utf8, &sb) == 0) {
    /* Returns true iff we check that this path is a regular file */
    return !!S_ISDIR(sb.st_mode);
  }
  /* We can't prove that given path is a file, so we return false. */
  return 0;
#elif KLC_OS_WINDOWS
  DWORD ftype = GetFileAttributesW(KLC_windows_get_wstr(path));
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
