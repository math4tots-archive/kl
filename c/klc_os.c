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

KLCNString* KLCNosZBnameZEinit() {
  return KLC_mkstr(KLC_OS_NAME);
}

KLCNString* KLCNosZBpathZBsepZEinit() {
  return KLC_mkstr(
    #if KLC_OS_WINDOWS
      "\\"
    #else
      "/"
    #endif
  );
}

KLCNTry* KLCNosZBtryChdir(KLCNString* path) {
  #if KLC_POSIX
    if (chdir(path->utf8) == 0) {
      return KLCNTryZEnew(1, KLC_int_to_var(0));
    } else {
      int errval = errno;
      return KLC_failm(strerror(errval));
    }
  #elif KLC_OS_WINDOWS
    if (SetCurrentDirectoryW(KLC_windows_get_wstr(path))) {
      return KLCNTryZEnew(1, KLC_int_to_var(0));
    } else {
      return KLC_failm("os.chdir failed");
    }
  #else
    /* TODO: Include KLC_OS_NAME in the error message */
    return KLC_failm("os.chdir not supported on this platform");
  #endif
}

KLCNTry* KLCNosZBtryMkdir(KLCNString* path) {
  #if KLC_POSIX
    if (mkdir(path->utf8, 0777) == 0) {
      return KLCNTryZEnew(1, KLC_int_to_var(0));
    } else {
      int errval = errno;
      return KLC_failm(strerror(errval));
    }
  #elif KLC_OS_WINDOWS
    if (CreateDirectoryW(KLC_windows_get_wstr(path), NULL)) {
      return KLCNTryZEnew(1, KLC_int_to_var(0));
    } else {
      return KLC_failm("os.mkdir failed");
    }
  #else
    /* TODO: Include KLC_OS_NAME in the error message */
    return KLC_failm("os.mkdir not supported on this platform");
  #endif
}

KLCNTry* KLCNosZBtryListdir(KLCNString* path) {
#if KLC_POSIX
  DIR* d;
  struct dirent* dir;
  KLCNList* ret = KLC_mklist(0);
  KLCNTry* t;
  d = opendir(path->utf8);
  if (!d) {
    int errval = errno;
    return KLC_failm(strerror(errval));
  }
  while ((dir = readdir(d)) != NULL) {
    KLC_header* dirname = (KLC_header*) KLC_mkstr(dir->d_name);
    KLCNListZFpush(ret, KLC_object_to_var(dirname));
    KLC_release(dirname);
  }
  closedir(d);
  t = KLCNTryZEnew(1, KLC_object_to_var((KLC_header*) ret));
  KLC_release((KLC_header*) ret);
  return t;
#elif KLC_OS_WINDOWS
  /* TODO: Use unicode versions of these functions */
  KLCNString* suffix = KLC_mkstr("\\*");
  KLCNString* pattern = KLCNStringZFAdd(path, suffix);
  WIN32_FIND_DATAW data;
  HANDLE hFind = FindFirstFileW(KLC_windows_get_wstr(pattern), &data);
  KLCNList* ret = KLC_mklist(0);
  if (hFind == INVALID_HANDLE_VALUE) {
    /* Read failed. TODO: Better error handling */
    return KLC_failm("os.listdir failed");
  }
  do {
    KLC_header* name = (KLC_header*) KLC_windows_string_from_wstr(data.cFileName);
    KLCNListZFpush(ret, KLC_object_to_var(name));
    KLC_release(name);
  } while (FindNextFileW(hFind, &data));
  FindClose(hFind);
  KLC_release((KLC_header*) suffix);
  KLC_release((KLC_header*) pattern);
  t = KLCNTryZEnew(1, KLC_object_to_var((KLC_header*) ret));
  KLC_release((KLC_header*) ret);
  return t;
#else
  /* TODO: Include KLC_OS_NAME in the error message */
  return KLC_failm("os.listdir not supported on this platform");
#endif
}

KLCNTry* KLCNosZBtryGetcwd() {
#if KLC_POSIX
  size_t cap = 2, len;
  char* buffer = (char*) malloc(sizeof(char) * cap);
  KLCNTry* t;
  KLCNString* s;
  while (getcwd(buffer, cap) == NULL) {
    int errval = errno;
    if (errval != ERANGE) {
      /* We failed to getcwd for a reason besides not big enough buffer
       * TODO: Better error handling */
      return KLC_failm(strerror(errval));
    }
    cap *= 2;
    buffer = (char*) realloc(buffer, sizeof(char) * cap);
  }
  len = strlen(buffer);
  buffer = (char*) realloc(buffer, sizeof(char) * (len + 1));
  s = KLC_mkstr_with_buffer(len, buffer, KLC_check_ascii(buffer));
  t = KLCNTryZEnew(1, KLC_object_to_var((KLC_header*) s));
  KLC_release((KLC_header*) s);
  return t;
#elif KLC_OS_WINDOWS
  DWORD bufsize;
  LPWSTR buf;
  KLCNTry* t;
  KLCNString* s;
  bufsize = GetCurrentDirectoryW(0, NULL) + 1;
  buf = (LPWSTR) malloc(bufsize * 2);
  GetCurrentDirectoryW(bufsize, buf);
  s = KLC_windows_string_from_wstr_buffer(buf);
  t = KLCNTryZEnew(1, KLC_object_to_var((KLC_header*) s));
  KLC_release((KLC_header*) s);
  return t;
#else
  /* TODO: Include KLC_OS_NAME in the error message */
  return KLC_failm("os.getcwd not supported for this platform");
#endif
}

KLC_bool KLCNosZBisfile(KLCNString* path) {
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

KLC_bool KLCNosZBisdir(KLCNString* path) {
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
