#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INVALID_TYPE     0
#define NIL_TYPE         1
#define BOOL_TYPE        2
#define SYMBOL_TYPE      3
#define NUMBER_TYPE      4
#define OBJECT_TYPE      5

#define STRING_TYPE      6
#define LIST_TYPE        7
#define SCOPE_TYPE       8
#define FUNCTION_TYPE    9

#define NIL (g.nil_value)
#define TRUE (g.true_value)
#define FALSE (g.false_value)

typedef size_t Symbol;
typedef int Type;
typedef struct Value Value;
typedef struct Object Object;
typedef struct String String;
typedef struct List List;
typedef struct Scope Scope;
typedef struct Function Function;
typedef Value (*Implementation)(void*, int, Value*);

struct Value {
  Type type;
  union {
    int b;
    Symbol s;
    double n;
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

struct Scope {
  Object header;
  Value parent;
  size_t size;
  size_t capacity;
  Value *buffer;
};

struct Function {
  Object header;
  char *name;
  Implementation implementation;
  Value data;
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

static const char *typeToName(Type t) {
  switch (t) {
    case INVALID_TYPE:
      return "INVALID";
    case NIL_TYPE:
      return "NIL";
    case BOOL_TYPE:
      return "BOOL";
    case SYMBOL_TYPE:
      return "SYMBOL";
    case NUMBER_TYPE:
      return "NUMBER";
    case OBJECT_TYPE:
      return "OBJECT";
    case STRING_TYPE:
      return "STRING";
    case LIST_TYPE:
      return "LIST";
    case SCOPE_TYPE:
      return "SCOPE";
    case FUNCTION_TYPE:
      return "FUNCTION";
  }
  errorf("typeToName: invalid type code: %d", t);
  return NULL;
}

static Type getType(Value v) {
  return v.type == OBJECT_TYPE ? v.u.o->type : v.type;
}

static const char *nameOfType(Value v) {
  return typeToName(getType(v));
}

static int isType(Value v, Type t) {
  return getType(v) == t;
}

static void expectType(const char *fname, Value v, Type t) {
  if (!isType(v, t)) {
    errorf("%s: expected type %s but got %s",
           fname, typeToName(t), nameOfType(v));
  }
}

static void retain(Value v) {
  if (v.type == OBJECT_TYPE) {
    v.u.o->refcnt++;
  }
}

static void release(Value v) {
  if (v.type == OBJECT_TYPE) {
    /* TODO: Do this in a way that won't cause stackoverflow */
    Object *o = v.u.o;
    if (o->refcnt > 1) {
      o->refcnt--;
    } else {
      switch (o->type) {
        case STRING_TYPE:
          free(((String*) o)->buffer);
          break;
        case LIST_TYPE:
          {
            size_t i;
            List *list = (List*) o;
            for (i = 0; i < list->size; i++) {
              release(list->buffer[i]);
            }
            free(list->buffer);
          }
          break;
        case SCOPE_TYPE:
          {
            size_t i;
            Scope *scope = (Scope*) o;
            release(scope->parent);
            for (i = 0; i < scope->capacity; i++) {
              if (scope->buffer[i].type != INVALID_TYPE) {
                release(scope->buffer[i]);
              }
            }
            free(scope->buffer);
          }
          break;
        case FUNCTION_TYPE:
          {
            Function *f = (Function*) o;
            free(f->name);
            release(f->data);
          }
          break;
        default:
          errorf("release: Invalid object type: %d", o->type);
      }
      free(o);
    }
  }
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

static void initObject(Object *object, Type type) {
  object->type = type;
  object->refcnt = 1;
}

static Value newString(const char *s) {
  Value ret;
  String *str;
  size_t len;

  len = strlen(s);
  str = (String*) malloc(sizeof(String));
  str->size = len;
  str->buffer = (char*) malloc(sizeof(char) * (len + 1));
  strcpy(str->buffer, s);

  ret.type = OBJECT_TYPE;
  ret.u.o = (Object*) str;
  initObject(ret.u.o, STRING_TYPE);

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

static Value newList() {
  Value ret;
  List *list;

  list = (List*) malloc(sizeof(List));
  list->size = list->capacity = 0;
  list->buffer = NULL;

  ret.type = OBJECT_TYPE;
  ret.u.o = (Object*) list;
  initObject(ret.u.o, LIST_TYPE);

  return ret;
}

static size_t ListSize(Value v) {
  expectType("ListSize", v, LIST_TYPE);
  return ((List*) v.u.o)->size;
}

static Value newScope(Value parent) {
  Value ret;
  Scope *scope;

  scope = (Scope*) malloc(sizeof(Scope));
  scope->size = scope->capacity = 0;
  scope->buffer = NULL;
  scope->parent = parent;

  retain(parent);

  ret.type = OBJECT_TYPE;
  ret.u.o = (Object*) scope;
  initObject(ret.u.o, SCOPE_TYPE);

  return ret;
}

static Value newFunction(const char *name, Implementation i, Value data) {
  Value ret;
  Function *f;

  f = (Function*) malloc(sizeof(Function));
  f->name = (char*) malloc(sizeof(char) * (strlen(name) + 1));
  strcpy(f->name, name);
  f->implementation = i;
  f->data = data;

  ret.type = OBJECT_TYPE;
  ret.u.o = (Object*) f;
  initObject(ret.u.o, FUNCTION_TYPE);

  return ret;
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
