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

void KLC_panic_with_error(KLC_Error* error) {
  fprintf(stderr, "ERROR: %s\n", error->message);
  exit(1);
}

KLC_Error* KLC_new_error_with_message(const char* msg) {
  KLC_Error* error = malloc(sizeof(KLC_Error));
  error->message = KLC_CopyString(msg);
  return error;
}

const char* KLC_get_error_message(KLC_Error* error) {
  return error->message;
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

void KLC_retain_var(KLC_var v) {
  if (v.tag == KLC_TAG_POINTER) {
    KLC_retain(v.u.p);
  }
}

void KLC_release_var(KLC_var v) {
  if (v.tag == KLC_TAG_POINTER) {
    KLC_release(v.u.p);
  }
}

void KLC_partial_release_var(KLC_var v, KLC_Header** delete_queue) {
  if (v.tag == KLC_TAG_POINTER) {
    KLC_partial_release(v.u.p, delete_queue);
  }
}

KLC_var KLC_var_from_ptr(KLC_Header* p) {
  KLC_var ret;
  ret.tag = KLC_TAG_POINTER;
  ret.u.p = p;
  return ret;
}

KLC_var KLC_var_from_int(KLC_int i) {
  KLC_var ret;
  ret.tag = KLC_TAG_INT;
  ret.u.i = i;
  return ret;
}

KLC_var KLC_var_from_float(KLC_float f) {
  KLC_var ret;
  ret.tag = KLC_TAG_FLOAT;
  ret.u.f = f;
  return ret;
}

KLC_Error* KLC_var_to_ptr(KLC_Header** out, KLC_var v, KLC_Class* cls) {
  if (v.tag != KLC_TAG_POINTER) {
    return KLC_new_error_with_message("Expected class type");
  }
  if (v.u.p->cls != cls) {
    return KLC_new_error_with_message("Not correct class type");
  }
  *out = v.u.p;
  return NULL;
}

KLC_Error* KLC_var_to_int(KLC_int* out, KLC_var v) {
  if (v.tag != KLC_TAG_INT) {
    return KLC_new_error_with_message("Expected integral type");
  }
  *out = v.u.i;
  return NULL;
}

KLC_Error* KLC_var_to_float(KLC_float* out, KLC_var v) {
  if (v.tag != KLC_TAG_FLOAT) {
    return KLC_new_error_with_message("Expected float type");
  }
  *out = v.u.f;
  return NULL;
}
