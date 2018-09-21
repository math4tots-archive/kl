#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NIL_TYPE         1
#define BOOL_TYPE        2
#define SYMBOL_TYPE      3
#define NUMBER_TYPE      4
#define FUNCTION_TYPE    5
#define OBJECT_TYPE      6

#define STRING_TYPE      7
#define LIST_TYPE        8

#define NIL (g.nil_value)
#define TRUE (g.true_value)
#define FALSE (g.false_value)

typedef size_t Symbol;
typedef int Type;
typedef struct Value Value;
typedef struct Object Object;
typedef struct String String;
typedef struct List List;
typedef Value (*Function)(int, Value*);

struct Value {
  Type type;
  union {
    int b;
    Symbol s;
    double n;
    Function f;
    Object *o;
  } u;
};

struct Object {
  Type type;
  size_t refcnt;
};

struct String {
  Object header;
  size_t size;
  char *buffer;
};

struct List {
  Object header;
  size_t size;
  size_t capacity;
  Value *buffer;
};

struct Globals {
  int initialized;

  long nsymbols;
  const char **symbols;

  Value nil_value;
  Value true_value;
  Value false_value;
} g;

static void errorf(const char *format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  fprintf(stderr, "\n");
  exit(1);
}

void init() {
  if (g.initialized) {
    return;
  }
  g.initialized = 1;

  g.nsymbols = 0;
  g.symbols = NULL;

  g.nil_value.type = NIL_TYPE;
  g.true_value.type = BOOL_TYPE;
  g.true_value.u.b = 1;
  g.false_value.type = BOOL_TYPE;
  g.false_value.u.b = 0;
}

static const char *nameOfType(Value v) {
  switch (v.type) {
    case NIL_TYPE:
      return "NIL";
    case BOOL_TYPE:
      return "BOOL";
    case SYMBOL_TYPE:
      return "SYMBOL";
    case NUMBER_TYPE:
      return "NUMBER";
    case FUNCTION_TYPE:
      return "FUNCTION";
    case OBJECT_TYPE:
      switch (v.u.o->type) {
        case STRING_TYPE:
          return "STRING";
        case LIST_TYPE:
          return "LIST";
        default:
          errorf("Invalid object type: %d", v.u.o->type);
      }
  }
  errorf("Invalid value type: %d", v.type);
  return NULL;
}

static Value newSymbol(const char *s) {
  Value ret;
  long i;
  char *buffer;

  ret.type = SYMBOL_TYPE;
  for (i = 0; i < g.nsymbols; i++) {
    if (strcmp(g.symbols[i], s) == 0) {
      ret.u.s = i;
      return ret;
    }
  }

  buffer = (char*) malloc(sizeof(char) * (strlen(s) + 1));
  strcpy(buffer, s);

  i = ++g.nsymbols;
  g.symbols = (const char**) realloc(g.symbols, g.nsymbols);
  g.symbols[i] = buffer;
  ret.u.s = i;
  return ret;
}

static Value newNumber(double n) {
  Value ret;
  ret.type = NUMBER_TYPE;
  ret.u.n = n;
  return ret;
}

static Value newFunction(Function f) {
  Value ret;
  ret.type = FUNCTION_TYPE;
  ret.u.f = f;
  return ret;
}

static void initObject(Object *object, Type type) {
  object->type = type;
  object->refcnt = 1;
}

static Value newString(const char *s) {
  Value ret;
  String *str;
  size_t len;

  len = strlen(s);
  ret.type = OBJECT_TYPE;
  str = (String*) malloc(sizeof(String));
  ret.u.o = (Object*) str;
  initObject(ret.u.o, STRING_TYPE);
  str->size = len;
  str->buffer = (char*) malloc(sizeof(char) * (len + 1));
  strcpy(str->buffer, s);

  return ret;
}

static Value newList() {
  Value ret;
  List *list;

  ret.type = OBJECT_TYPE;
  list = (List*) malloc(sizeof(List));
  list->size = list->capacity = 0;
  list->buffer = NULL;
  ret.u.o = (Object*) list;

  return ret;
}

static const char *cstr(Value v) {
  if (v.type == SYMBOL_TYPE) {
    return g.symbols[v.u.s];
  }
  if (v.type == OBJECT_TYPE && v.u.o->type == STRING_TYPE) {
    return ((String*) v.u.o)->buffer;
  }

  errorf("cstr: expected SYMBOL or STRING but got %s", nameOfType(v));
  return NULL;
}

int main() {
  Value v;
  init();

  v = NIL;
  printf("%d\n", v.type);

  v = newSymbol("Hello");

  printf("cstr(:Hello) = %s\n", cstr(v));

  /* printf("cstr(NIL) = %s\n", cstr(NIL)); */

  return 0;
}
