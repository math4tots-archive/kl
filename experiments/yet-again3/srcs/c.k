#"no builtins"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef * size_t

extern c_int printf(c_char const* msg, ...)
extern void* malloc(size_t size)
extern void free(void* p)
extern size_t strlen(c_char const* str)

extern c_char* cmalloc(int size) {
  static_cast(c_char*, malloc(static_cast(size_t, size)))
}

extern void cfree(c_char* p) {
  free(static_cast(void*, p))
}
