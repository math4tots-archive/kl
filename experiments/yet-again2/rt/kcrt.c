#include "kcrt.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

struct KLC_Error {
  char* message;
};

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

const char* KLC_get_error_message(void* errorp) {
  return ((KLC_Error*) errorp)->message;
}

void KLC_retain(KLC_Header* obj) {
  if (obj) {
    obj->refcnt++;
  }
}

void KLC_release(KLC_Header* obj) {
  KLC_Header* delete_queue = NULL;
  KLC_partial_release(obj, &delete_queue);
  while (delete_queue) {
    obj = delete_queue;
    printf("obj = %p\n", (void*) obj);
    printf("obj->cls = %p\n", (void*) obj->cls);
    delete_queue = delete_queue->next;
    obj->cls->deleter(obj, &delete_queue);
    free(obj);
  }
}

void KLC_partial_release(KLC_Header* obj, KLC_Header** delete_queue) {
  if (obj) {
    if (obj->refcnt) {
      obj->refcnt--;
    } else {
      obj->next = *delete_queue;
      *delete_queue = obj;
    }
  }
}
