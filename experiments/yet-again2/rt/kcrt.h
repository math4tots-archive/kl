#ifndef kcrt_h
#define kcrt_h
#include <stddef.h>

char* KLC_CopyString(const char* s);
void KLC_panic_with_error(void* errorp);
void* KLC_new_error_with_message(const char*);
const char* KLC_get_error_message(void*);

#endif/*kcrt_h*/
