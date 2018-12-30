#include "kl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MALLOC(type) ((type*) malloc(sizeof(type)))
#define MALLOC_STR(len) ((char*) malloc(sizeof(char) * (len)))
#define UNUSED(x) (void)(x)  /* For suppressing gcc's unused param warn */
#define CAST(x, st) (*((st*) (x).u.pointer))

typedef struct KL_String KL_String;
typedef struct KL_List KL_List;
typedef struct KL_Function KL_Function;
typedef struct KL_Builtin KL_Builtin;
typedef struct KL_Scope KL_Scope;
typedef struct KL_Source KL_Source;
typedef struct KL_Token KL_Token;
typedef struct KL_Ast KL_Ast;
typedef struct KL_Literal KL_Literal;
typedef struct KL_GetVar KL_GetVar;
typedef struct KL_SetVar KL_SetVar;
typedef struct KL_Block KL_Block;
typedef struct KL_FunctionCall KL_FunctionCall;
typedef struct KL_If KL_If;
typedef struct KL_While KL_While;
typedef struct KL_FunctionDisplay KL_FunctionDisplay;

struct KL {
  KL_Value nilv;

  /* TODO: Stack trace */
  /* TODO: Keep a linked list of all live retainables,
     and delete them all when this struct is deleted */
};

struct KL_Retainable {
  size_t refcnt;
};

struct KL_String {
  KL_Retainable header;
  size_t size;
  char *buffer;
};

struct KL_List {
  KL_Retainable header;
  size_t size, cap;
  KL_Value *buffer;
};

struct KL_Function {
  KL_Retainable header;
  KL_Value scope; /* Scope */
  KL_Value ast; /* FunctionDisplay */
};

struct KL_Builtin {
  KL_Retainable header;
  KL_Value name; /* String */
  KL_BF *bf;
};

struct KL_Scope {
  KL_Retainable header;
  KL_Value parent; /* Scope */
  size_t size, cap;
  char **keys;
  KL_Value *values;
};

struct KL_Source {
  KL_Retainable header;
  KL_Value path; /* String */
  KL_Value data; /* String */
};

struct KL_Token {
  KL_Retainable header;
  KL_Value source; /* Source */
  size_t i;
  KL_Value type; /* String */
  KL_Value value; /* String or number or nil */
};

struct KL_Ast {
  KL_Retainable header;
  KL_Value token; /* Token */
};

struct KL_Literal {
  KL_Ast header;
  KL_Value value;
};

struct KL_GetVar {
  KL_Ast header;
  int initialized;
  char *name;
  KL_Index index;
};

struct KL_SetVar {
  KL_Ast header;
  int initialized;
  char *name;
  KL_Index index;
  KL_Value expr; /* Ast */
};

struct KL_Block {
  KL_Ast header;
  KL_Value exprs; /* List of Ast */
};

struct KL_FunctionCall {
  KL_Ast header;
  KL_Value fexpr; /* Ast */
  KL_Value argexprs; /* List of Ast */
};

struct KL_If {
  KL_Ast header;
  KL_Value cond; /* Ast */
  KL_Value body; /* Ast */
  KL_Value other; /* Ast */
};

struct KL_While {
  KL_Ast header;
  KL_Value cond; /* Ast */
  KL_Value body; /* Ast */
};

struct KL_FunctionDisplay {
  KL_Ast header;
  KL_Value name; /* String */
  KL_Value argnames; /* List of String */
  KL_Value body; /* Ast */
};

static void KL_init_retainable(KL *kl, KL_Retainable *r) {
  UNUSED(kl);
  r->refcnt = 0;
}

static void KL_init_ast(KL *kl, KL_Ast *r, KL_Value token) {
  KL_init_retainable(kl, (KL_Retainable*) r);
  KL_retain(kl, token);
  r->token = token;
}

static void KL_assert_list_bound(KL *kl, KL_Value list, size_t i) {
  size_t size = KL_list_size(kl, list);
  KL_assert(kl, i < size);
}

static void KL_free(KL *kl, KL_Value value) {
  UNUSED(value);
  KL_panic_with_message(kl, "TODO: KL_free");
}

KL *KL_new() {
  KL *kl = MALLOC(KL);
  kl->nilv.type = KL_NIL;
  return kl;
}

void KL_delete(KL *kl) {
  free(kl);
}

void KL_panic(KL *kl) {
  /* TODO: Dump stack trace */
  int one = 1, zero = 0, result;
  KL_delete(kl);
  result = one / zero; /* Try to trigger segfault */
  fprintf(stderr, "PANIC %d\n", result);
  exit(1); /* If triggering segfault fails, just exit */
}

void KL_panic_with_message(KL *kl, const char *msg) {
  fprintf(stderr, "PANIC: %s\n", msg);
  KL_panic(kl);
}

const char *KL_type_str(int type) {
  switch (type) {
    case KL_NIL:
      return "NIL";
    case KL_NUMBER:
      return "NUMBER";
    case KL_STRING:
      return "STRING";
    case KL_LIST:
      return "LIST";
    case KL_FUNCTION:
      return "FUNCTION";
    case KL_BUILTIN:
      return "BUILTIN";
    case KL_SCOPE:
      return "SCOPE";
    case KL_SOURCE:
      return "SOURCE";
    case KL_TOKEN:
      return "TOKEN";
    case KL_LITERAL:
      return "LITERAL";
    case KL_GET_VAR:
      return "GET_VAR";
    case KL_SET_VAR:
      return "SET_VAR";
    case KL_BLOCK:
      return "BLOCK";
    case KL_FUNCTION_CALL:
      return "FUNCTION_CALL";
    case KL_IF:
      return "IF";
    case KL_WHILE:
      return "WHILE";
    case KL_FUNCTION_DISPLAY:
      return "FUNCTION_DISPLAY";
  }
  return "INVALID";
}

int KL_is_nil(KL *kl, KL_Value value) {
  UNUSED(kl);
  return value.type == KL_NIL;
}

int KL_is_type(KL *kl, KL_Value value, int type) {
  UNUSED(kl);
  return value.type == type;
}

int KL_is_type_ast(KL *kl, KL_Value value) {
  UNUSED(kl);
  return value.type >= KL_AST_START && value.type < KL_AST_END;
}

int KL_is_retainable(KL *kl, KL_Value value) {
  UNUSED(kl);
  return value.type >= KL_RETAINABLE_START;
}

void KL_assert(KL *kl, int cond) {
  if (!cond) {
    fprintf(stderr, "Assertion failed\n");
    KL_panic(kl);
  }
}

void KL_assert_type(KL *kl, KL_Value value, int type) {
  if (!KL_is_type(kl, value, type)) {
    fprintf(
      stderr,
      "Expected type %s but got %s\n",
      KL_type_str(type),
      KL_type_str(value.type));
    KL_panic(kl);
  }
}

void KL_assert_type_ast(KL *kl, KL_Value value) {
  if (!KL_is_type_ast(kl, value)) {
    fprintf(
      stderr,
      "Expected Ast type, but got %s\n",
      KL_type_str(value.type));
    KL_panic(kl);
  }
}

void KL_assert_type_list_of(KL *kl, KL_Value value, int type) {
  KL_assert_type(kl, value, KL_LIST);
  {
    size_t size = KL_list_size(kl, value);
    size_t i = 0;
    for (; i < size; i++) {
      KL_Value item = KL_list_get(kl, value, i);
      KL_assert_type(kl, item, type);
      KL_release(kl, item);
    }
  }
}

void KL_assert_type_list_of_ast(KL *kl, KL_Value value) {
  KL_assert_type(kl, value, KL_LIST);
  {
    size_t size = KL_list_size(kl, value);
    size_t i = 0;
    for (; i < size; i++) {
      KL_Value item = KL_list_get(kl, value, i);
      KL_assert_type_ast(kl, item);
      KL_release(kl, item);
    }
  }
}

KL_Value KL_nil(KL *kl) {
  return kl->nilv;
}

KL_Value KL_new_number(KL *kl, double value) {
  KL_Value ret;
  UNUSED(kl);
  ret.type = KL_NUMBER;
  ret.u.number = value;
  return ret;
}

KL_Value KL_new_string(KL *kl, const char *cstr) {
  KL_Value ret;
  KL_String *s = MALLOC(KL_String);
  size_t len = strlen(cstr);
  ret.type = KL_STRING;
  ret.u.pointer = (KL_Retainable*) s;
  KL_init_retainable(kl, ret.u.pointer);
  s->size = len;
  s->buffer = MALLOC_STR(len + 1);
  strcpy(s->buffer, cstr);
  return ret;
}

KL_Value KL_new_list(KL *kl) {
  KL_Value ret;
  KL_List *list = MALLOC(KL_List);
  ret.type = KL_LIST;
  ret.u.pointer = (KL_Retainable*) list;
  KL_init_retainable(kl, ret.u.pointer);
  list->size = list->cap = 0;
  list->buffer = NULL;
  return ret;
}

KL_Value KL_new_function(KL *kl, KL_Value scope, KL_Value ast) {
  KL_Value ret;
  KL_Function *f = MALLOC(KL_Function);
  KL_assert_type(kl, scope, KL_SCOPE);
  KL_assert_type_ast(kl, ast);
  ret.type = KL_FUNCTION;
  ret.u.pointer = (KL_Retainable*) f;
  KL_init_retainable(kl, ret.u.pointer);
  KL_retain(kl, scope);
  KL_retain(kl, ast);
  f->scope = scope;
  f->ast = ast;
  return ret;
}

KL_Value KL_new_builtin(KL *kl, KL_Value name, KL_BF *bf) {
  KL_Value ret;
  KL_Builtin *b = MALLOC(KL_Builtin);
  KL_assert_type(kl, name, KL_STRING);
  ret.type = KL_BUILTIN;
  ret.u.pointer = (KL_Retainable*) b;
  KL_init_retainable(kl, ret.u.pointer);
  KL_retain(kl, name);
  b->name = name;
  b->bf = bf;
  return ret;
}

KL_Value KL_new_scope(KL *kl, KL_Value parent) {
  KL_Value ret;
  KL_Scope *s = MALLOC(KL_Scope);
  if (!KL_is_nil(kl, parent)) {
    KL_assert_type(kl, parent, KL_SCOPE);
  }
  ret.type = KL_SCOPE;
  ret.u.pointer = (KL_Retainable*) s;
  KL_init_retainable(kl, ret.u.pointer);
  KL_retain(kl, parent);
  s->parent = parent;
  s->size = s->cap = 0;
  s->keys = NULL;
  s->values = NULL;
  return ret;
}

KL_Value KL_new_source(KL *kl, KL_Value path, KL_Value data) {
  KL_Value ret;
  KL_Source *s = MALLOC(KL_Source);
  KL_assert_type(kl, path, KL_STRING);
  KL_assert_type(kl, data, KL_STRING);
  ret.type = KL_SOURCE;
  ret.u.pointer = (KL_Retainable*) s;
  KL_init_retainable(kl, ret.u.pointer);
  KL_retain(kl, path);
  KL_retain(kl, data);
  s->path = path;
  s->data = data;
  return ret;
}

KL_Value KL_new_token(
    KL *kl, KL_Value source, size_t i, KL_Value type, KL_Value value) {
  KL_Value ret;
  KL_Token *t = MALLOC(KL_Token);
  KL_assert_type(kl, source, KL_SOURCE);
  KL_assert_type(kl, type, KL_STRING);
  ret.type = KL_TOKEN;
  ret.u.pointer = (KL_Retainable*) t;
  KL_init_retainable(kl, ret.u.pointer);
  KL_retain(kl, source);
  KL_retain(kl, type);
  KL_retain(kl, value);
  t->source = source;
  t->i = i;
  t->type = type;
  t->value = value;
  return ret;
}

KL_Value KL_new_literal(KL *kl, KL_Value token, KL_Value value) {
  KL_Value ret;
  KL_Literal *lit = MALLOC(KL_Literal);
  KL_assert_type(kl, token, KL_TOKEN);
  ret.type = KL_LITERAL;
  ret.u.pointer = (KL_Retainable*) lit;
  KL_init_ast(kl, (KL_Ast*) lit, token);
  KL_retain(kl, value);
  lit->value = value;
  return ret;
}

KL_Value KL_new_get_var(KL*, KL_Value, const char*);
KL_Value KL_new_set_var(KL*, KL_Value, const char*, KL_Value);
KL_Value KL_new_block(KL*, KL_Value, KL_Value);
KL_Value KL_new_function_call(KL*, KL_Value, KL_Value, KL_Value);
KL_Value KL_new_if(KL*, KL_Value, KL_Value, KL_Value, KL_Value);
KL_Value KL_new_while(KL*, KL_Value, KL_Value, KL_Value);
KL_Value KL_new_function_display(KL*, KL_Value, KL_Value, KL_Value, KL_Value);

void KL_retain(KL *kl, KL_Value value) {
  if (KL_is_retainable(kl, value)) {
    value.u.pointer->refcnt++;
  }
}

void KL_release(KL *kl, KL_Value value) {
  if (KL_is_retainable(kl, value)) {
    if (value.u.pointer->refcnt) {
      value.u.pointer->refcnt--;
    } else {
      KL_free(kl, value);
    }
  }
  if (!KL_is_retainable(kl, value)) {
    return;
  }
  if (value.u.pointer->refcnt) {
    value.u.pointer->refcnt--;
    return;
  }
  KL_panic_with_message(kl, "TODO: KL_release");
}

double KL_number_get(KL*, KL_Value);
const char *KL_string_get(KL*, KL_Value);

size_t KL_list_size(KL *kl, KL_Value list) {
  KL_assert_type(kl, list, KL_LIST);
  return CAST(list, KL_List).size;
}

KL_Value KL_list_get(KL *kl, KL_Value list, size_t i) {
  KL_Value ret;
  KL_assert_list_bound(kl, list, i);
  ret = CAST(list, KL_List).buffer[i];
  KL_retain(kl, ret);
  return ret;
}

void KL_list_set(KL*, KL_Value, size_t, KL_Value);
void KL_list_push(KL*, KL_Value, KL_Value);
KL_Value KL_list_pop(KL*, KL_Value);
KL_Value KL_function_invoke(KL*, KL_Value, KL_Value);
KL_Value KL_builtin_invoke(KL*, KL_Value, KL_Value);
size_t KL_scope_size(KL*, KL_Value);
int KL_scope_find(KL*, KL_Value, const char*, KL_Index*);
KL_Value KL_scope_fast_get(KL*, KL_Value, KL_Index);
void KL_scope_fast_set(KL*, KL_Value, KL_Index, KL_Value);
KL_Value KL_scope_get(KL*, KL_Value, const char*);
KL_Value KL_scope_set(KL*, KL_Value, const char*, KL_Value);
KL_Value KL_invoke(KL*, KL_Value, KL_Value);
int KL_truthy(KL*, KL_Value);

KL_Value KL_lex(KL *kl, KL_Value source) {
  KL_Value tokens = KL_new_list(kl);
  UNUSED(kl);
  UNUSED(source);
  KL_panic_with_message(kl, "TODO: KL_lex");
  return tokens;
}

KL_Value KL_parse(KL*, KL_Value);

KL_Value KL_eval(KL *kl, KL_Value scope, KL_Value ast) {
  UNUSED(kl);
  UNUSED(scope);
  UNUSED(ast);
  KL_panic_with_message(kl, "TODO: KL_eval");
  return KL_nil(kl);
}

