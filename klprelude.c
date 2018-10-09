/* Autogenerated by the KL Compiler */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KLC_MAX_STACK_SIZE 1000
#define KLC_TAG_BOOL 0
#define KLC_TAG_INT 1
#define KLC_TAG_DOUBLE 2
#define KLC_TAG_FUNCTION 3
#define KLC_TAG_TYPE 4
#define KLC_TAG_OBJECT 5

typedef struct KLC_stack_frame KLC_stack_frame;
typedef char KLC_bool;
typedef long KLC_int;
typedef struct KLC_header KLC_header;
typedef struct KLC_methodinfo KLC_methodinfo;
typedef struct KLC_methodlist KLC_methodlist;
typedef struct KLC_typeinfo KLC_typeinfo;
typedef struct KLC_var KLC_var;
typedef KLC_var (*KLC_fp)(int, KLC_var*); /* untyped function pointer */
typedef struct KLC_functioninfo KLC_functioninfo;
typedef KLC_functioninfo* KLC_function;
typedef KLC_typeinfo* KLC_type;
typedef struct KLCNString KLCNString;
typedef struct KLCNList KLCNList;


struct KLC_stack_frame {
  const char* filename;
  const char* function;
  long lineno;
};

struct KLC_header {
  KLC_typeinfo* type;
  size_t refcnt;
  KLC_header* next;
};

struct KLC_methodinfo {
  const char* const name;
  const KLC_fp body;
};

struct KLC_methodlist {
  const size_t size;
  const KLC_methodinfo* const methods;
};

struct KLC_functioninfo {
  const char* const name;
  const KLC_fp body;
};

struct KLC_typeinfo {
  const char* const name;
  void (*const deleter)(KLC_header*, KLC_header**);
  const KLC_methodlist* const methods;
};

struct KLC_var {
  int tag;
  union {
    KLC_header* obj;
    KLC_int i;
    double d;
    KLC_function f;
    KLC_type t;
  } u;
};

void KLC_errorf(const char* fmt, ...);
KLCNString* KLC_mkstr_with_buffer(size_t bytesize, char* str, int is_ascii);
KLCNString* KLC_mkstr(const char *str);
KLC_var KLC_mcall(const char* name, int argc, KLC_var* argv);
KLC_bool KLC_truthy(KLC_var v);
void KLC_release(KLC_header *obj);
KLC_bool KLC_var_to_bool(KLC_var v);

size_t KLC_stacktrace_size = 0;
KLC_stack_frame KLC_stacktrace[KLC_MAX_STACK_SIZE];

size_t KLC_release_on_exit_buffer_cap = 0;
size_t KLC_release_on_exit_buffer_size = 0;
KLC_header** KLC_release_on_exit_buffer = NULL;

void KLC_push_frame(const char* filename, const char* function, long ln) {
  KLC_stacktrace[KLC_stacktrace_size].filename = filename;
  KLC_stacktrace[KLC_stacktrace_size].function = function;
  KLC_stacktrace[KLC_stacktrace_size].lineno = ln;
  KLC_stacktrace_size++;
  if (KLC_stacktrace_size == KLC_MAX_STACK_SIZE) {
    KLC_errorf("stackoverflow");
  }
}

void KLC_pop_frame() {
  KLC_stacktrace_size--;
}

const char* KLC_tag_to_string(int tag) {
  switch (tag) {
    case KLC_TAG_BOOL: return "BOOL";
    case KLC_TAG_INT: return "INT";
    case KLC_TAG_DOUBLE: return "DOUBLE";
    case KLC_TAG_FUNCTION: return "FUNCTION";
    case KLC_TAG_TYPE: return "TYPE";
    case KLC_TAG_OBJECT: return "OBJECT";
  }
  KLC_errorf("tag_to_string: invalid tag %d", tag);
  return NULL;
}

void KLC_release_object_on_exit(KLC_header* obj) {
  if (KLC_release_on_exit_buffer_size >= KLC_release_on_exit_buffer_cap) {
    KLC_release_on_exit_buffer_cap += 10;
    KLC_release_on_exit_buffer_cap *= 2;
    KLC_release_on_exit_buffer =
      (KLC_header**) realloc(
        KLC_release_on_exit_buffer,
        sizeof(KLC_header*) * KLC_release_on_exit_buffer_cap);
  }
  KLC_release_on_exit_buffer[KLC_release_on_exit_buffer_size++] = obj;
}

void KLC_release_var_on_exit(KLC_var v) {
  if (v.tag == KLC_TAG_OBJECT) {
    KLC_release_object_on_exit(v.u.obj);
  }
}

void KLC_release_queued_before_exit() {
  size_t i = 0;
  for (; i < KLC_release_on_exit_buffer_size; i++) {
    KLC_release(KLC_release_on_exit_buffer[i]);
  }
  free(KLC_release_on_exit_buffer);
}

void KLC_errorf(const char* fmt, ...) {
  size_t i;
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fprintf(stderr, "\n");
  for (i = 0; i < KLC_stacktrace_size; i++) {
    KLC_stack_frame *frame = KLC_stacktrace + i;
    fprintf(
      stderr,
      "  %s %s line %lu\n",
      frame->function,
      frame->filename,
      frame->lineno);
  }
  exit(1);
}

void KLC_retain(KLC_header *obj) {
  if (obj) {
    obj->refcnt++;
  }
}

void KLC_retain_var(KLC_var v) {
  if (v.tag == KLC_TAG_OBJECT) {
    KLC_retain(v.u.obj);
  }
}

void KLC_partial_release(KLC_header* obj, KLC_header** delete_queue) {
  if (obj) {
    if (obj->refcnt) {
      obj->refcnt--;
    } else {
      obj->next = *delete_queue;
      *delete_queue = obj;
    }
  }
}

void KLC_partial_release_var(KLC_var v, KLC_header** delete_queue) {
  if (v.tag == KLC_TAG_OBJECT) {
    KLC_partial_release(v.u.obj, delete_queue);
  }
}

void KLC_release(KLC_header *obj) {
  KLC_header* delete_queue = NULL;
  KLC_partial_release(obj, &delete_queue);
  while (delete_queue) {
    obj = delete_queue;
    delete_queue = delete_queue->next;
    obj->type->deleter(obj, &delete_queue);
    free(obj);
  }
}

void KLC_release_var(KLC_var v) {
  if (v.tag == KLC_TAG_OBJECT) {
    KLC_release(v.u.obj);
  }
}

extern KLC_typeinfo KLC_typenull;
extern KLC_typeinfo KLC_typebool;
extern KLC_typeinfo KLC_typeint;
extern KLC_typeinfo KLC_typedouble;
extern KLC_typeinfo KLC_typefunction;
extern KLC_typeinfo KLC_typetype;

KLC_bool KLCN_Is(KLC_var a, KLC_var b) {
  if (a.tag != b.tag) {
    return 0;
  }
  switch (a.tag) {
    case KLC_TAG_BOOL:
    case KLC_TAG_INT:
      return a.u.i == b.u.i;
    case KLC_TAG_DOUBLE:
      return a.u.d == b.u.d;
    case KLC_TAG_FUNCTION:
      return a.u.f == b.u.f;
    case KLC_TAG_TYPE:
      return a.u.t == b.u.t;
    case KLC_TAG_OBJECT:
      return a.u.obj == b.u.obj;
  }
  KLC_errorf("_Is: invalid tag: %d", a.tag);
  return 0;
}

KLC_bool KLCN_IsNot(KLC_var a, KLC_var b) {
  return !KLCN_Is(a, b);
}

KLC_bool KLCN_Eq(KLC_var a, KLC_var b) {
  if (a.tag == b.tag) {
    switch (a.tag) {
      case KLC_TAG_BOOL:
        return !!a.u.i == !!b.u.i;
      case KLC_TAG_INT:
        return a.u.i == b.u.i;
      case KLC_TAG_DOUBLE:
        return a.u.d == b.u.d;
      case KLC_TAG_FUNCTION:
        return a.u.f == b.u.f;
      case KLC_TAG_TYPE:
        return a.u.t == b.u.t;
      case KLC_TAG_OBJECT:
        if (a.u.obj == b.u.obj) {
          return 1;
        } else if (a.u.obj && b.u.obj && a.u.obj->type == b.u.obj->type) {
          KLC_var args[2];
          KLC_var r;
          KLC_bool br;
          args[0] = a;
          args[1] = b;
          r = KLC_mcall("Eq", 2, args);
          br = KLC_truthy(r);
          KLC_release_var(r);
          return br;
        }
      default:
        KLC_errorf("_Eq: invalid tag: %d", a.tag);
    }
  } else if (a.tag == KLC_TAG_INT && b.tag == KLC_TAG_DOUBLE) {
    return a.u.i == b.u.d;
  } else if (a.tag == KLC_TAG_DOUBLE && b.tag == KLC_TAG_INT) {
    return a.u.d == b.u.i;
  }
  return 0;
}

KLC_bool KLCN_Ne(KLC_var a, KLC_var b) {
  return !KLCN_Eq(a, b);
}

KLC_bool KLCN_Lt(KLC_var a, KLC_var b) {
  if (a.tag == b.tag) {
    switch (a.tag) {
      case KLC_TAG_INT:
        return a.u.i < b.u.i;
      case KLC_TAG_DOUBLE:
        return a.u.d < b.u.d;
    }
  } else if (a.tag == KLC_TAG_INT && b.tag == KLC_TAG_DOUBLE) {
    return a.u.i < b.u.d;
  } else if (a.tag == KLC_TAG_DOUBLE && b.tag == KLC_TAG_INT) {
    return a.u.d < b.u.i;
  }
  {
    KLC_var args[2];
    args[0] = a;
    args[1] = b;
    return KLC_var_to_bool(KLC_mcall("Lt", 2, args));
  }
}

KLC_bool KLCN_Gt(KLC_var a, KLC_var b) {
  return KLCN_Lt(b, a);
}

KLC_bool KLCN_Le(KLC_var a, KLC_var b) {
  return !KLCN_Lt(b, a);
}

KLC_bool KLCN_Ge(KLC_var a, KLC_var b) {
  return !KLCN_Lt(a, b);
}

KLC_bool KLCNbool(KLC_var v) {
  return KLC_truthy(v);
}

KLC_type KLCNtype(KLC_var v) {
  switch (v.tag) {
    case KLC_TAG_BOOL:
      return &KLC_typebool;
    case KLC_TAG_INT:
      return &KLC_typeint;
    case KLC_TAG_DOUBLE:
      return &KLC_typedouble;
    case KLC_TAG_FUNCTION:
      return &KLC_typefunction;
    case KLC_TAG_TYPE:
      return &KLC_typetype;
    case KLC_TAG_OBJECT:
      return v.u.obj ? v.u.obj->type : &KLC_typenull;
  }
  KLC_errorf("Unrecognized type tag %d", v.tag);
  return NULL;
}

KLC_var KLC_bool_to_var(KLC_bool b) {
  KLC_var ret;
  ret.tag = KLC_TAG_BOOL;
  ret.u.i = b;
  return ret;
}

KLC_var KLC_int_to_var(KLC_int i) {
  KLC_var ret;
  ret.tag = KLC_TAG_INT;
  ret.u.i = i;
  return ret;
}

KLC_var KLC_double_to_var(double d) {
  KLC_var ret;
  ret.tag = KLC_TAG_DOUBLE;
  ret.u.d = d;
  return ret;
}

KLC_var KLC_function_to_var(KLC_function f) {
  KLC_var ret;
  ret.tag = KLC_TAG_FUNCTION;
  ret.u.f = f;
  return ret;
}

KLC_var KLC_type_to_var(KLC_type t) {
  KLC_var ret;
  ret.tag = KLC_TAG_TYPE;
  ret.u.t = t;
  return ret;
}

KLC_var KLC_object_to_var(KLC_header* obj) {
  KLC_var ret;
  ret.tag = KLC_TAG_OBJECT;
  ret.u.obj = obj;
  return ret;
}

KLC_bool KLC_var_to_bool(KLC_var v) {
  /* TODO: Better error message */
  if (v.tag != KLC_TAG_BOOL) {
    KLC_errorf("var_to_bool: expected bool (tag %d) but got tag %d",
               KLC_TAG_BOOL, v.tag);
  }
  return v.u.i ? 1 : 0;
}

KLC_bool KLC_truthy(KLC_var v) {
  switch (v.tag) {
    case KLC_TAG_BOOL:
    case KLC_TAG_INT:
      return v.u.i;
    case KLC_TAG_DOUBLE:
      return v.u.d;
    case KLC_TAG_FUNCTION:
    case KLC_TAG_TYPE:
      return 1;
    case KLC_TAG_OBJECT:
      return KLC_var_to_bool(KLC_mcall("Bool", 1, &v));
  }
  KLC_errorf("truthy: invalid tag %d", v.tag);
  return 0;
}

KLC_int KLC_var_to_int(KLC_var v) {
  /* TODO: Better error message */
  if (v.tag != KLC_TAG_INT) {
    KLC_errorf("var_to_int: expected int (tag %d) but got tag %d",
               KLC_TAG_INT, v.tag);
  }
  return v.u.i;
}

double KLC_var_to_double(KLC_var v) {
  /* TODO: Better error message */
  if (v.tag != KLC_TAG_DOUBLE) {
    KLC_errorf("var_to_double: expected double (tag %d) but got tag %d",
               KLC_TAG_DOUBLE, v.tag);
  }
  return v.u.d;
}

KLC_function KLC_var_to_function(KLC_var v) {
if (v.tag != KLC_TAG_FUNCTION) {
  KLC_errorf("var_to_type: expected function (tag %d) but got tag %d",
             KLC_TAG_FUNCTION, v.tag);
}
return v.u.f;
}

KLC_type KLC_var_to_type(KLC_var v) {
  /* TODO: Better error message */
  if (v.tag != KLC_TAG_TYPE) {
    KLC_errorf("var_to_type: expected type (tag %d) but got tag %d",
               KLC_TAG_TYPE, v.tag);
  }
  return v.u.t;
}

KLC_header* KLC_var_to_object(KLC_var v, KLC_type ti) {
  /* TODO: Better error message */
  KLC_header* ret;
  if (v.tag != KLC_TAG_OBJECT) {
    KLC_errorf("var_to_object: not an object");
  }
  if (ti != KLCNtype(v)) {
    KLC_errorf("var_to_object: not the right object type");
  }
  return v.u.obj;
}

KLC_var KLC_var_call(KLC_var f, int argc, KLC_var* argv) {
  /* TODO: Better error message */
  KLC_var result;

  if (f.tag != KLC_TAG_FUNCTION) {
    KLC_errorf("Not a function");
  }

  result = f.u.f->body(argc, argv);

  return result;
}

KLC_var KLC_mcall(const char* name, int argc, KLC_var* argv) {
  if (argc == 0) {
    KLC_errorf("mcall requires at least 1 arg");
  }
  {
    KLC_type type = KLCNtype(argv[0]);
    const KLC_methodlist* mlist = type->methods;
    size_t len = mlist->size;
    const KLC_methodinfo* mbuf = mlist->methods;
    const KLC_methodinfo* m = NULL;
    size_t i;
    /* TODO: Faster method dispatch mechanism */
    if (len) {
      int cmp = strcmp(name, mbuf[0].name);
      if (cmp == 0) {
        m = mbuf;
      } else if (cmp > 0) {
        size_t lower = 0;
        size_t upper = len;
        while (lower + 1 < upper) {
          size_t mid = (lower + upper) / 2;
          cmp = strcmp(name, mbuf[mid].name);
          if (cmp == 0) {
            m = mbuf + mid;
            break;
          } else if (cmp < 0) {
            upper = mid;
          } else {
            lower = mid;
          }
        }
      }
    }
    if (!m) {
      KLC_errorf("No such method '%s' for type '%s'", name, type->name);
    }
    return m->body(argc, argv);
  }
}

KLC_int KLCNint_mAdd(KLC_int a, KLC_int b) {
  return a + b;
}

KLC_int KLCNint_mSub(KLC_int a, KLC_int b) {
  return a - b;
}

KLC_bool KLCNint_mEq(KLC_int a, KLC_int b) {
  return a == b;
}

KLC_bool KLCNint_mNe(KLC_int a, KLC_int b) {
  return a != b;
}

KLC_bool KLCNint_mLt(KLC_int a, KLC_int b) {
  return a < b;
}

KLCNString* KLCNint_mRepr(KLC_int i) {
  char buffer[50];
  sprintf(buffer, "%ld", i);
  return KLC_mkstr(buffer);
}

double KLCNdouble_mAdd(double a, double b) {
  return a + b;
}

double KLCNdouble_mSub(double a, double b) {
  return a - b;
}

KLC_bool KLCNdouble_mEq(double a, double b) {
  return a == b;
}

KLC_bool KLCNdouble_mNe(double a, double b) {
  return a != b;
}

KLC_bool KLCNdouble_mLt(double a, double b) {
  return a < b;
}

KLCNString* KLCNdouble_mRepr(double d) {
  char buffer[80];
  sprintf(buffer, "%f", d);
  return KLC_mkstr(buffer);
}

KLCNString* KLCNfunction_mGETname(KLC_function f) {
  return KLC_mkstr(f->name);
}

KLCNString* KLCNtype_mGETname(KLC_type t) {
  return KLC_mkstr(t->name);
}

KLC_bool KLCNtype_mEq(KLC_type a, KLC_type b) {
  return a == b;
}

void KLC_init_header(KLC_header* header, KLC_type type) {
  header->type = type;
  header->refcnt = 0;
  header->next = NULL;
}

struct KLCNString {
  /* TODO: Also keep a UTF32 representation for random access */
  KLC_header header;
  size_t bytesize; /* number of actual bytes in buffer (utf-8 representation) */
  size_t nchars;   /* number of unicode code points */
  char* buffer;
  char* utf32;
    /* In C89, there's no way to get an integer type that guarantees
     * exactly 32-bits. As such, I want to error on side of correctness and
     * use chars instead for measuring out the utf-32 representation.
     */
  int is_ascii;
};

void KLC_deleteString(KLC_header* robj, KLC_header** dq) {
  KLCNString* obj = (KLCNString*) robj;
  free(obj->buffer);
}

extern KLC_typeinfo KLC_typeString;

int KLC_check_ascii(const char* str) {
  while (*str) {
    if (((unsigned) *str) >= 128) {
      return 0;
    }
    str++;
  }
  return 1;
}

KLCNString* KLC_mkstr_with_buffer(size_t bytesize, char* str, int is_ascii) {
  KLCNString* obj = (KLCNString*) malloc(sizeof(KLCNString));
  if (!is_ascii) {
    KLC_errorf("Non-ascii strings not yet supported");
  }
  KLC_init_header(&obj->header, &KLC_typeString);
  obj->bytesize = bytesize;
  obj->nchars = bytesize; /* only true when is_ascii */
  obj->buffer = str;
  obj->utf32 = NULL;
  obj->is_ascii = is_ascii;
  return obj;
}

KLCNString* KLC_mkstr(const char *str) {
  size_t len = strlen(str);
  char* buffer = (char*) malloc(sizeof(char) * (len + 1));
  strcpy(buffer, str);
  return KLC_mkstr_with_buffer(len, buffer, KLC_check_ascii(buffer));
}

KLC_int KLCNString_mbytesize(KLCNString* s) {
  /* TODO: This may be lossy. Figure this out */
  return (KLC_int) s->bytesize;
}

KLC_int KLCNString_mGETsize(KLCNString* s) {
  /* TODO: This may be lossy. Figure this out */
  return (KLC_int) s->nchars;
}

KLCNString* KLCNString_mStr(KLCNString* s) {
  KLC_retain((KLC_header*) s);
  return s;
}

KLCNString* KLCNString_mAdd(KLCNString* a, KLCNString* b) {
  size_t bytesize = a->bytesize + b->bytesize;
  char* buffer = (char*) malloc(sizeof(char) * (bytesize + 1));
  strcpy(buffer, a->buffer);
  strcpy(buffer + a->bytesize, b->buffer);
  return KLC_mkstr_with_buffer(bytesize, buffer, a->is_ascii && b->is_ascii);
}

KLC_bool KLCNString_mEq(KLCNString* a, KLCNString* b) {
  return strcmp(a->buffer, b->buffer) == 0;
}

KLC_bool KLCNString_mLt(KLCNString* a, KLCNString* b) {
  return strcmp(a->buffer, b->buffer) < 0;
}

struct KLCNList {
  KLC_header header;
  size_t size, cap;
  KLC_var* buffer;
};

extern KLC_typeinfo KLC_typeList;

KLCNList* KLC_mklist(size_t cap) {
  KLCNList* obj = (KLCNList*) malloc(sizeof(KLCNList));
  KLC_init_header(&obj->header, &KLC_typeList);
  obj->size = 0;
  obj->cap = cap;
  obj->buffer = cap ? (KLC_var*) malloc(sizeof(KLC_var) * cap) : NULL;
  return obj;
}

void KLC_deleteList(KLC_header* robj, KLC_header** dq) {
  KLCNList* list = (KLCNList*) robj;
  size_t i;
  for (i = 0; i < list->size; i++) {
    KLC_partial_release_var(list->buffer[i], dq);
  }
  free(list->buffer);
}

void KLCNList_mpush(KLCNList* list, KLC_var v) {
  if (list->size >= list->cap) {
    list->cap += 4;
    list->cap *= 2;
    list->buffer = (KLC_var*) realloc(list->buffer, sizeof(KLC_var) * list->cap);
  }
  list->buffer[list->size++] = v;
  KLC_retain_var(v);
}

KLC_var KLCNList_mGetItem(KLCNList* list, KLC_int i) {
  if (i < 0) {
    i += list->size;
  }
  if (i < 0 || ((size_t) i) >= list->size) {
    KLC_errorf("Index out of bounds (i = %ld, size = %ld)", i, list->size);
  }
  KLC_retain_var(list->buffer[i]);
  return list->buffer[i];
}

KLC_var KLCNList_mSetItem(KLCNList* list, KLC_int i, KLC_var v) {
  if (i < 0) {
    i += list->size;
  }
  if (i < 0 || ((size_t) i) >= list->size) {
    KLC_errorf("Index out of bounds (i = %ld, size = %ld)", i, list->size);
  }
  list->buffer[i] = v;
  KLC_retain_var(v);  /* once for attaching value to list */
  KLC_retain_var(v);  /* once more for using as return value */
  return v;
}

KLC_int KLCNList_mGETsize(KLCNList* list) {
  return (KLC_int) list->size;
}

void KLCNputs(KLCNString* s) {
  printf("%s\n", s->buffer);
}

void KLCNassert(KLC_var cond) {
  if (!KLC_truthy(cond)) {
    KLC_errorf("Assertion failed");
  }
}

void KLCNmain();

static const KLC_var KLC_null = {
  KLC_TAG_OBJECT,
  { NULL }
};

int main() {
  KLCNmain();
  KLC_release_queued_before_exit();
  return 0;
}
