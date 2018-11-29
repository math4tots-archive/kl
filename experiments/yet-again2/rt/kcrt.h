#ifndef kcrt_h
#define kcrt_h
#include <stddef.h>

typedef struct KLC_Error KLC_Error;
typedef struct KLC_Header KLC_Header;
typedef struct KLC_Class KLC_Class;
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

char* KLC_CopyString(const char* s);
void KLC_panic_with_error(void* errorp);
void* KLC_new_error_with_message(const char*);
const char* KLC_get_error_message(void*);

void KLC_retain(KLC_Header*);
void KLC_release(KLC_Header*);
void KLC_partial_release(KLC_Header*, KLC_Header**);

#endif/*kcrt_h*/
