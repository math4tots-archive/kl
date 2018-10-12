#ifndef klc_os_h
#define klc_os_h
#include "klc_prelude.h"

/* I want to eventually support at least POSIX and Windows.
 * Anything not yet supported will cause the 'os' global variable
 * set to null.
 */
#if __APPLE__

#define KLC_OS_NAME "darwin"
#define KLC_POSIX 1
#define KLC_OS_APPLE 1

  #if TARGET_OS_IPHONE && TARGET_IPHONE_SIMULATOR
    /* simulator */
  #elif TARGET_OS_IPHONE
    /* iPhone */
  #else
    /* Not iPhone */
  #endif

#elif __ANDROID__

#define KLC_OS_NAME "android"

/* OK, android is known for being not exactly posix, but for many
 * things it's good enough */
#define KLC_POSIX 1
#define KLC_OS_ANDROID 1

#elif __linux__

#define KLC_OS_NAME "linux"
#define KLC_POSIX 1
#define KLC_OS_LINUX 1

#elif defined(_WIN32)

#define KLC_OS_NAME "windows"
#define KLC_OS_WINDOWS 1

#elif defined(_POSIX_C_SOURCE) || defined(_POSIX_VERSION)

#define KLC_OS_NAME "posix"
#define KLC_POSIX 1
#define KLC_OS_POSIX 1

#else

#define KLC_OS_NAME "unknown"
#define KLC_OS_UNKNOWN 1

#endif


typedef struct KLCNOperatingSystemInterface KLCNOperatingSystemInterface;

struct KLCNOperatingSystemInterface {
  KLC_header header;
};

extern KLC_typeinfo KLC_typeOperatingSystemInterface;

KLCNOperatingSystemInterface* KLCN_initos();
void KLC_deleteOperatingSystemInterface(KLC_header* robj, KLC_header** dq);
KLCNString* KLCNOperatingSystemInterface_mGETname(KLCNOperatingSystemInterface*);
KLC_bool KLCNOperatingSystemInterface_mGETposix(KLCNOperatingSystemInterface*);
KLC_bool KLCNOperatingSystemInterface_mBool(KLCNOperatingSystemInterface*);
KLCNList* KLCNOperatingSystemInterface_mlistdir(
  KLCNOperatingSystemInterface*,
  KLCNString*);

#endif/*klc_os_h*/
