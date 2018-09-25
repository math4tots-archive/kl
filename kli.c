/*
KL interpreter

cls && \
gcc -Werror -Wpedantic -Wall -Wno-unused-function kli.c && \
cp kli.{c,cc} \
&& gcc -Werror -Wpedantic -Wall -Wno-unused-function kli.cc && \
./a.out && \
rm a.out kli.cc
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define OK 0
#define ERR 1

typedef struct Object Object;
typedef struct FieldEntry FieldEntry;
typedef struct Globals Globals;
typedef int Status;

struct Object {
  size_t refcnt;
  Object *next; /* linked list used to delete objects */
  Object *proto;

  size_t nfields;
  FieldEntry *fields;

  /* These fields could perhaps be a union,
   * but not making them a union makes it easier
   * to delete objects */
  double n; /* for number */
  size_t size; /* for bool, string and list */
  size_t capacity; /* for list */
  char *s; /* for string */
  Object **a; /* for list */
  Status (*f)(Object* out, size_t argc, Object **argv); /* for function */
};

struct FieldEntry {
  char *key;
  Object *value;
};

struct Globals {
  int initialized;
  Object *nil;
  Object *bool_proto;
  Object *tru;
  Object *fal;
  Object *number_proto;
  Object *string_proto;
  Object *list_proto;
};

static Object *mkobj(Object*);

static Globals g;

static void init() {
  if (g.initialized) {
    return;
  }
  g.initialized = 1;
  g.nil = mkobj(NULL);
  g.bool_proto = mkobj(NULL);
  g.tru = mkobj(g.bool_proto);
  g.tru->size = 1;
  g.fal = mkobj(g.bool_proto);
  g.fal->size = 0;
  g.number_proto = mkobj(NULL);
  g.string_proto = mkobj(NULL);
  g.list_proto = mkobj(NULL);
}

static Object *retain(Object *obj) {
  if (obj) {
    obj->refcnt++;
  }
  return obj;
}

static void partial_release(Object *obj, Object **delete_queue) {
  if (obj->refcnt <= 1) {
    obj->next = *delete_queue;
    *delete_queue = obj;
  } else {
    obj->refcnt--;
  }
}

static void release(Object *obj) {
  if (obj) {
    Object *delete_queue = NULL;
    partial_release(obj, &delete_queue);
    while (delete_queue) {
      size_t i;
      obj = delete_queue;
      delete_queue = obj->next;

      for (i = 0; i < obj->nfields; i++) {
        partial_release(obj->fields[i].value, &delete_queue);
      }
      free(obj->s);
      if (obj->a) {
        for (i = 0; i < obj->size; i++) {
          partial_release(obj->a[i], &delete_queue);
        }
        free(obj->a);
      }

      free(obj);
    }
  }
}

static Object *mkobj(Object *proto) {
  Object *ret = (Object*) malloc(sizeof(Object));
  ret->refcnt = 1;
  ret->next = NULL;
  ret->proto = proto;
  ret->nfields = 0;
  ret->fields = NULL;
  ret->n = 0;
  ret->size = ret->capacity = 0;
  ret->s = NULL;
  ret->a = NULL;
  ret->f = NULL;
  return ret;
}

static char *copystr(const char *str) {
  size_t len = strlen(str);
  char *ret = (char*) malloc(sizeof(char) * (len + 1));
  strcpy(ret, str);
  return ret;
}

static Object *mkstr(const char *str) {
  Object *ret = mkobj(g.string_proto);
  ret->size = ret->capacity = strlen(str);
  ret->s = copystr(str);
  return ret;
}

static FieldEntry *findattr(Object *obj, const char *attr) {
  size_t i;
  for (i = 0; i < obj->nfields; i++) {
    if (strcmp(obj->fields[i].key, attr) == 0) {
      return obj->fields + i;
    }
  }
  return NULL;
}

static Object *getattr(Object *obj, const char *attr) {
  FieldEntry *entry = findattr(obj, attr);
  return entry ? retain(entry->value) : NULL;
}

static void setattr(Object *obj, const char *attr, Object *value) {
  FieldEntry *entry = findattr(obj, attr);
  retain(value);
  if (entry) {
    release(entry->value);
    entry->value = value;
  } else {
    obj->nfields++;
    obj->fields =
      (FieldEntry*) realloc(obj->fields, sizeof(FieldEntry) * obj->nfields);
    obj->fields[obj->nfields - 1].key = copystr(attr);
    obj->fields[obj->nfields - 1].value = value;
  }
}

int main() {
  init();
}
