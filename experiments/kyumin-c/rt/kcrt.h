#ifndef KCRT_H
#define KCRT_H
/* KC runtime */
/* Assumes UNIVERSAL_PREFIX = 'KLC' */
#include <limits.h>
#include <stddef.h>

#define KLC_TAG_POINTER 0
#define KLC_TAG_BOOL 1
#define KLC_TAG_INT 2
#define KLC_TAG_FLOAT 3

#if defined(LLONG_MAX)
typedef long long KLCint;
#define KLC_INT_FMT "%lld"
#define KLC_INT_MAX LLONG_MAX
#define KLC_INT_MIN LLONG_MIN
#else
typedef long KLCint;
#define KLC_INT_FMT "%ld"
#define KLC_INT_MAX LONG_MAX
#define KLC_INT_MIN LONG_MIN
#endif

/* TODO: Also compare with PTRDIFF_MAX */
#if KLC_INT_MAX <= 2147483647L
#error "32-bit long (want at least 64 bits)"
#endif

typedef double KLCfloat;
typedef struct KLCvar KLCvar;
typedef struct KLCheader KLCheader;
typedef struct KLCXClass KLCXClass;
typedef struct KLCXAutoReleasePool KLCXAutoReleasePool;
typedef KLCvar KLCXMethod(KLCvar, int, KLCvar*);

struct KLCvar {
  int tag;
  union {
    KLCheader* p;
    KLCint i;
    KLCfloat f;
  } u;
};

struct KLCheader {
  size_t refcnt;
  KLCXClass* cls;
};

struct KLCXAutoReleasePool {
  size_t size, cap;
  KLCvar* buffer;
};

/* OOP utilities */
KLCXClass* KLCXGetClass(KLCvar);
const char* KLCXGetClassName(KLCXClass*);
KLCXMethod* KLCXGetMethodForClass(KLCXClass*, const char*);
KLCvar KLCXCallMethod(KLCvar, const char*, int, ...);
KLCXClass* KLCXNewClass(const char*);
KLCvar KLCXAddMethod(KLCXClass*, const char*, KLCXMethod*);

/* Reference counting utilities */
KLCvar KLCXPush(KLCXAutoReleasePool*, KLCvar);
void KLCXResize(KLCXAutoReleasePool*, size_t);

#endif/*KCRT_H*/
