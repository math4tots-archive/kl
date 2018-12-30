#include "kl.h"
#include <stdio.h>

typedef struct KL_String KL_String;
typedef struct KL_List KL_List;
typedef struct KL_Function KL_Function;
typedef struct KL_Builtin KL_Builtin;
typedef struct KL_Scope KL_Scope;
typedef struct KL_Ast KL_Ast;
typedef struct KL_Literal KL_Literal;
typedef struct KL_Block KL_Block;
typedef struct KL_FunctionCall KL_FunctionCall;

struct KL {
  KL_Value nilv;

  /* TODO: Stack trace */
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
  KL_Value data; /* String or number or nil */
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
