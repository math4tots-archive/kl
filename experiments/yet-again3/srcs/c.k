#"no builtins"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern c_int printf(c_char const* msg, ...)
extern void* malloc(size_t size)
extern void free(void* p)
extern void* realloc(void* p, size_t size)
extern size_t strlen(c_char const* str)
extern c_int sprintf(c_char* str, c_char const* fmt, ...)

extern c_char* cmalloc(int size) {
  dynamic_cast(c_char*, malloc(dynamic_cast(size_t, size)))
}

extern void cfree(c_char* p) {
  free(dynamic_cast(void*, p))
}

extern c_char* crealloc(c_char* p, int size) {
  dynamic_cast(
    c_char*,
    realloc(
      dynamic_cast(void*, p),
      dynamic_cast(size_t, size)))
}

extern c_char* strcpy(c_char* dest, c_char const* src)
extern c_char* strcat(c_char* dest, c_char const* src)
