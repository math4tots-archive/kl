#include "kcrt.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>

typedef struct {
  char* name;
  KLCXMethod* method;
} KLCXMethodEntry;

struct KLCXClass {
  KLCheader header;
  char* name;
  KLCXDeleter* deleter;
  size_t nmethods;
  KLCXMethodEntry* methods;
};

const KLCvar KLCnull = {KLC_TAG_POINTER, {NULL}};

/* General C utilities */
char* KLCXCopyString(const char* s) {
  char* copy = malloc(sizeof(char) * (strlen(s) + 1));
  strcpy(copy, s);
  return copy;
}

/* OOP utilities */
void KLCXinit(KLCheader* obj, KLCXClass* cls) {
  obj->refcnt = 0;
  obj->next = NULL;
  obj->cls = cls;
}

KLCXClass* KLCXGetClass(KLCvar v) {
  if (v.tag == KLC_TAG_POINTER) {
    if (v.u.p) {
      return v.u.p->cls;
    }
  }
  assert(0); /* TODO */
  return NULL;
}

const char* KLCXGetClassName(KLCXClass* cls) {
  return cls->name;
}

KLCXMethod* KLCXGetMethodForClass(KLCXClass* cls, const char* name) {
  size_t i;
  for (i = 0; i < cls->nmethods; i++) {
    if (strcmp(cls->methods[i].name, name) == 0) {
      return cls->methods[i].method;
    }
  }
  return NULL;
}

KLCvar KLCXCallMethod(KLCvar owner, const char* name, int argc, ...) {
  KLCvar v;
  assert(0); /* TODO */
  return v;
}

static void initClass(KLCXClass* cls) {
  cls->nmethods = 0;
  cls->methods = NULL;
}

static void classDeleter(KLCheader* obj, KLCheader** dq) {
  KLCXClass* cls = (KLCXClass*) obj;
  size_t i;
  for (i = 0; i < cls->nmethods; i++) {
    free(cls->methods[i].name);
  }
  free(cls->methods);
}

KLCXClass* KLCXGetClassClass() {
  static KLCXClass* classClass = NULL;
  if (classClass == NULL) {
    classClass = malloc(sizeof(KLCXClass));
    KLCXinit((KLCheader*) classClass, classClass);
    classClass->name = KLCXCopyString("Class");
    classClass->deleter = classDeleter;
    initClass(classClass);
  }
  return classClass;
}

KLCXClass* KLCXNewClass(const char* name, KLCXDeleter* deleter) {
  KLCXClass* cls = malloc(sizeof(KLCXClass));
  KLCXinit((KLCheader*) cls, KLCXGetClassClass());
  cls->name = KLCXCopyString(name);
  cls->deleter = deleter;
  initClass(cls);
  return cls;
}

KLCXDeleter* KLCXGetDeleter(KLCXClass* cls) {
  return cls->deleter;
}

void KLCXAddMethod(KLCXClass* cls, const char* name, KLCXMethod* method) {
  size_t i;
  for (i = 0; i < cls->nmethods; i++) {
    if (strcmp(cls->methods[i].name, name)) {
      cls->methods[i].method = method;
      return;
    }
  }
  cls->nmethods++;
  cls->methods = realloc(cls->methods, sizeof(KLCXMethodEntry) * (cls->nmethods));
  cls->methods[cls->nmethods - 1].name = KLCXCopyString(name);
  cls->methods[cls->nmethods - 1].method = method;
}

KLCvar KLCXObjectToVar(KLCheader* obj) {
  KLCvar ret;
  ret.tag = KLC_TAG_POINTER;
  ret.u.p = obj;
  return ret;
}

/* Reference counting utilities */
KLCvar KLCXPush(KLCXReleasePool *pool, KLCvar v) {
  if (v.tag == KLC_TAG_POINTER && v.u.p) {
    if (pool->size >= pool->cap) {
      pool->cap = 2 * pool->cap + 8;
      pool->buffer =
        (KLCvar*) realloc(pool->buffer, sizeof(KLCvar) * pool->cap);
    }
    pool->buffer[pool->size++] = v;
  }
  return v;
}

void KLCXResize(KLCXReleasePool *pool, size_t size) {
  while (pool->size > size) {
    KLCXRelease(pool->buffer[--pool->size]);
  }
}

void KLCXDrainPool(KLCXReleasePool *pool) {
  KLCXResize(pool, 0);
  free(pool->buffer);
  pool->buffer = NULL;
}

void KLCXRetain(KLCvar v) {
  if (v.tag == KLC_TAG_POINTER && v.u.p) {
    v.u.p->refcnt++;
  }
}

void KLCXReleasePointer(KLCheader* obj) {
  KLCheader* delete_queue = NULL;
  KLCXPartialReleasePointer(obj, &delete_queue);
  while (delete_queue) {
    obj = delete_queue;
    delete_queue = delete_queue->next;
    obj->cls->deleter(obj, &delete_queue);
    free(obj);
  }
}

void KLCXRelease(KLCvar v) {
  if (v.tag == KLC_TAG_POINTER && v.u.p) {
    KLCXReleasePointer(v.u.p);
  }
}

void KLCXPartialRelease(KLCvar v, KLCheader** delete_queue) {
  if (v.tag == KLC_TAG_POINTER) {
    KLCXPartialReleasePointer(v.u.p, delete_queue);
  }
}

void KLCXPartialReleasePointer(KLCheader* obj, KLCheader** delete_queue) {
  if (obj) {
    if (obj->refcnt) {
      obj->refcnt--;
    } else {
      obj->next = *delete_queue;
      *delete_queue = obj;
    }
  }
}
