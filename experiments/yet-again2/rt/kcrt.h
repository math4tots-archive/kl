#ifndef kcrt_h
#define kcrt_h
#include <stddef.h>

#define KLC_TAG_POINTER 0
#define KLC_TAG_INT 1
#define KLC_TAG_FLOAT 2

typedef ptrdiff_t KLC_int;  /* TODO: This is unsatisfactory */
typedef double KLC_float;
typedef struct KLC_Error KLC_Error;
typedef struct KLC_Header KLC_Header;
typedef struct KLC_Class KLC_Class;
typedef struct KLC_var KLC_var;
typedef void KLC_Deleter(KLC_Header*, KLC_Header**);

struct KLC_Header {
  size_t refcnt;
  KLC_Class* cls;
  KLC_Header* next; /* for stackless release with partial_release */
};

struct KLC_Class {
  const char*const module_name;
  const char*const short_name;
  KLC_Deleter*const deleter;
};

struct KLC_var {
  int tag;
  union {
    KLC_Header* p;
    KLC_int i;
    KLC_float f;
  } u;
};

char* KLC_CopyString(const char* s);
void KLC_panic_with_error(KLC_Error* error);
KLC_Error* KLC_new_error_with_message(const char*);
const char* KLC_get_error_message(KLC_Error*);

void KLC_retain(KLC_Header*);
void KLC_release(KLC_Header*);
void KLC_partial_release(KLC_Header*, KLC_Header**);
void KLC_retain_var(KLC_var);
void KLC_release_var(KLC_var);
void KLC_partial_release_var(KLC_var, KLC_Header**);

KLC_var KLC_var_from_ptr(KLC_Header* p);
KLC_var KLC_var_from_int(KLC_int i);
KLC_var KLC_var_from_float(KLC_float f);
KLC_Error* KLC_var_to_ptr(KLC_Header** out, KLC_var v, KLC_Class*);
KLC_Error* KLC_var_to_int(KLC_int* out, KLC_var v);
KLC_Error* KLC_var_to_float(KLC_float* out, KLC_var v);

#endif/*kcrt_h*/
