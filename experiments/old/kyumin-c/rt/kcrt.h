#ifndef KCRT_H
#define KCRT_H
/* KC runtime */
/* Assumes UNIVERSAL_PREFIX = 'KLC' */
#include <limits.h>
#include <stddef.h>

#define KLC_TAG_POINTER 0 /* Needed so that KLCvar calloc is KLCnull */
#define KLC_TAG_BOOL 1
#define KLC_TAG_INT 2
#define KLC_TAG_FLOAT 3

#if 0 /* Always use just long for now */
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
typedef struct KLCXReleasePool KLCXReleasePool;
typedef KLCvar KLCXMethod(int, KLCvar*);
typedef KLCheader* KLCXAllocator();
typedef void KLCXDeleter(KLCheader*, KLCheader**);

struct KLCvar {
  int tag;
  union {
    KLCheader* p; /* This must come first, for 'null' initializing vars */
    KLCint i;
    KLCfloat f;
  } u;
};

struct KLCheader {
  size_t refcnt;
  KLCheader* next;
  KLCXClass* cls;
};

struct KLCXReleasePool {
  size_t size, cap;
  KLCvar* buffer;
};

extern const KLCvar KLCnull;

/* General C utilities */
char* KLCXCopyString(const char*);

/* OOP utilities */
void KLCXinit(KLCheader*, KLCXClass*);
KLCXClass* KLCXGetClass(KLCvar);
const char* KLCXGetClassName(KLCXClass*);
KLCXMethod* KLCXGetMethodForClass(KLCXClass*, const char*);
KLCvar KLCXCallMethod(const char*, int, ...);
KLCXClass* KLCXGetClassClass();
KLCXClass* KLCXNewClass(const char*, KLCXAllocator*, KLCXDeleter*);
KLCXDeleter* KLCXGetDeleter(KLCXClass*);
void KLCXAddMethod(KLCXClass*, const char*, KLCXMethod*);
KLCvar KLCXObjectToVar(KLCheader*);

/* Reference counting utilities */
KLCvar KLCXPush(KLCXReleasePool*, KLCvar);
void KLCXResize(KLCXReleasePool*, size_t);
void KLCXDrainPool(KLCXReleasePool*);
void KLCXRetain(KLCvar);
void KLCXReleasePointer(KLCheader*);
void KLCXRelease(KLCvar);
void KLCXPartialRelease(KLCvar, KLCheader**);
void KLCXPartialReleasePointer(KLCheader*, KLCheader**);

#endif/*KCRT_H*/
