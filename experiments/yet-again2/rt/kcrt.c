#include "kcrt.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct {
  char* message;
} KLC_Error;

char* KLC_CopyString(const char* s) {
  char* ret = (char*) malloc(sizeof(char) * (strlen(s) + 1));
  strcpy(ret, s);
  return ret;
}

void KLC_panic_with_error(void* errorp) {
  KLC_Error* error = (KLC_Error*) errorp;
  fprintf(stderr, "ERROR: %s\n", error->message);
  exit(1);
}

void* KLC_new_error_with_message(const char* msg) {
  KLC_Error* error = malloc(sizeof(KLC_Error));
  error->message = KLC_CopyString(msg);
  return error;
}

const char* KLC_get_error_message(const void* errorp) {
  return ((KLC_Error*) errorp)->message;
}
