#ifndef klc_plat_h
#define klc_plat_h

/* I want to eventually support at least POSIX and Windows.
 * Anything not yet supported will cause the 'os' global variable
 * set to null.
 *
 * Even if _POSIX_VERSION is defined, the minimum version of
 * of 200112 must be met to be considered 'POSIX'.
 * This corresponds to Issue 6.
 *
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

#elif _POSIX_VERSION >= 200112

#define KLC_OS_NAME "posix"
#define KLC_POSIX 1
#define KLC_OS_POSIX 1

#else

#define KLC_OS_NAME "unknown"
#define KLC_OS_UNKNOWN 1

#endif

#endif/*klc_plat_h*/
