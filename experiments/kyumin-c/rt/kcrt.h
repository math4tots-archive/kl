#ifndef KCRT_H
#define KCRT_H
/* KC runtime */
/* Assumes UNIVERSAL_PREFIX = 'KLC' */
#include <limits.h>
#include <stddef.h>

typedef long KLCint;
#define KLC_INT_FMT "%ld"
#define KLC_INT_MAX LONG_MAX

/* TODO: Also compare with PTRDIFF_MAX */
#if KLC_INT_MAX <= 2147483647L
#error "32-bit long (want at least 64 bits)"
#endif

typedef double KLCfloat;
typedef struct KLCvar KLCvar;

struct KLCvar {
  int tag;
  union {
    KLCint i;
    KLCfloat f;
  } u;
};

#endif/*KCRT_H*/
