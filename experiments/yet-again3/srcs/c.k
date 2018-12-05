#"no builtins"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef * size_t

extern c_int printf(c_char const* msg, ...)
extern void* malloc(size_t size)
extern void free(void* p)
extern void* realloc(void* p, size_t size)
extern size_t strlen(c_char const* str)
extern c_int sprintf(c_char* str, c_char const* fmt, ...)

extern c_char* cmalloc(int size) {
  static_cast(c_char*, malloc(static_cast(size_t, size)))
}

extern void cfree(c_char* p) {
  free(static_cast(void*, p))
}

extern c_char* crealloc(c_char* p, int size) {
  static_cast(
    c_char*,
    realloc(
      static_cast(void*, p),
      static_cast(size_t, size)))
}

extern c_char* strcpy(c_char* dest, c_char const* src)
extern c_char* strcat(c_char* dest, c_char const* src)
