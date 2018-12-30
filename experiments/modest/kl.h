#ifndef kl_h
#define kl_h
#include <stddef.h>

#define KL_NIL 0
#define KL_NUMBER 1
#define KL_STRING 100
#define KL_LIST 101
#define KL_FUNCTION 102
#define KL_BUILTIN 103
#define KL_SCOPE 104
#define KL_SOURCE 200
#define KL_TOKEN 201
#define KL_LITERAL 300
#define KL_GET_VAR 301
#define KL_SET_VAR 302
#define KL_BLOCK 303
#define KL_FUNCTION_CALL 304
#define KL_IF 305
#define KL_WHILE 306
#define KL_FUNCTION_DISPLAY 307

#define KL_RETAINABLE_START 100
#define KL_AST_START 300
#define KL_AST_END 400

typedef struct KL KL;
typedef struct KL_Value KL_Value;
typedef struct KL_Index KL_Index;
typedef struct KL_Retainable KL_Retainable;
typedef KL_Value KL_BF(KL*, KL_Value);

struct KL_Value {
  int type;
  union {
    double number;
    KL_Retainable *pointer;
  } u;
};

struct KL_Index {  /* For fast Scope lookup */
  size_t depth;
  size_t index;
};

KL *KL_new();
void KL_delete(KL*);
void KL_panic(KL*);
void KL_panic_with_message(KL*, const char*);
const char *KL_type_str(int);
int KL_is_nil(KL*, KL_Value);
int KL_is_type(KL*, KL_Value, int);
int KL_is_type_ast(KL*, KL_Value);
int KL_is_retainable(KL*, KL_Value);
void KL_assert(KL*, int);
void KL_assert_type(KL*, KL_Value, int);
void KL_assert_type_ast(KL*, KL_Value);
void KL_assert_type_list_of(KL*, KL_Value, int);
void KL_assert_type_list_of_ast(KL*, KL_Value);
KL_Value KL_nil(KL*);
KL_Value KL_new_number(KL*, double);
KL_Value KL_new_string(KL*, const char*);
KL_Value KL_new_list(KL*);
KL_Value KL_new_function(KL*, KL_Value, KL_Value);
KL_Value KL_new_builtin(KL*, KL_Value, KL_BF*);
KL_Value KL_new_scope(KL*, KL_Value);
KL_Value KL_new_source(KL*, KL_Value, KL_Value);
KL_Value KL_new_token(KL*, KL_Value, size_t, KL_Value, KL_Value);
KL_Value KL_new_literal(KL*, KL_Value, KL_Value);
KL_Value KL_new_get_var(KL*, KL_Value, const char*);
KL_Value KL_new_set_var(KL*, KL_Value, const char*, KL_Value);
KL_Value KL_new_block(KL*, KL_Value, KL_Value);
KL_Value KL_new_function_call(KL*, KL_Value, KL_Value, KL_Value);
KL_Value KL_new_if(KL*, KL_Value, KL_Value, KL_Value, KL_Value);
KL_Value KL_new_while(KL*, KL_Value, KL_Value, KL_Value);
KL_Value KL_new_function_display(KL*, KL_Value, KL_Value, KL_Value, KL_Value);
void KL_retain(KL*, KL_Value);
void KL_release(KL*, KL_Value);
double KL_number_get(KL*, KL_Value);
const char *KL_string_get(KL*, KL_Value);
size_t KL_list_size(KL*, KL_Value);
KL_Value KL_list_get(KL*, KL_Value, size_t);
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
KL_Value KL_parse(KL*, KL_Value);
KL_Value KL_eval(KL*, KL_Value);

#endif/*kl_h*/
