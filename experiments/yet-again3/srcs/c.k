#"no builtins"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef * size_t

extern int printf(char const* msg, ...)
extern void* malloc(size_t size)
extern void free(void* p)
extern size_t strlen(char const* str)
