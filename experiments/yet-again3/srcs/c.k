#"no builtins"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef * size_t

extern c_int printf(c_char const* msg, ...)
extern void* malloc(size_t size)
extern void free(void* p)
extern size_t strlen(c_char const* str)
