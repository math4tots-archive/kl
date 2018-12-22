#ifndef kcrt_h
#define kcrt_h
#include <stddef.h>

#define KLC_TAG_OBJECT 0
#define KLC_TAG_BOOL 1
#define KLC_TAG_INT 2
#define KLC_TAG_FLOAT 3
#define KLC_TAG_TYPE 4

typedef int KLC_bool;
typedef ptrdiff_t KLC_int;  /* TODO: This is unsatisfactory */
typedef double KLC_float;
typedef struct KLC_Error KLC_Error;
typedef struct KLC_Stack KLC_Stack;
typedef struct KLC_Header KLC_Header;
typedef struct KLC_MethodEntry KLC_MethodEntry;
typedef struct KLC_Class KLC_Class;
typedef struct KLC_var KLC_var;
typedef struct KLC_Lambda_capture KLC_Lambda_capture;
typedef void KLC_Deleter(KLC_Header*, KLC_Header**);
typedef KLC_Error* KLC_Method(KLC_Stack*, KLC_var*, int, KLC_var*);
typedef KLC_Error* KLC_Lambda_body(
  KLC_Stack*, KLC_var*, KLC_Lambda_capture*, int, KLC_var*);

struct KLC_Header {
  size_t refcnt;
  KLC_Class* cls;
  KLC_Header* next; /* for stackless release with partial_release */
};

struct KLC_MethodEntry {
  const char*const name;
  KLC_Method* method;
};

struct KLC_Class {
  const char*const module_name;
  const char*const short_name;
  KLC_Deleter*const deleter;
  const size_t number_of_methods;
  KLC_MethodEntry* methods;
  const size_t number_of_class_methods;
  KLC_MethodEntry* class_methods;
};

struct KLC_var {
  int tag;
  union {
    KLC_Header* p;
    KLC_bool b;
    KLC_int i;
    KLC_float f;
    KLC_Class* t;
  } u;
};

struct KLC_Lambda_capture {
  size_t size;
  KLC_var* buffer;
};

extern KLC_var KLC_null;
extern KLC_Class KLC_type_class;
extern KLC_Class KLC_int_class;

char* KLC_CopyString(const char* s);

KLC_Stack* KLC_new_stack();
void KLC_delete_stack(KLC_Stack*);
void KLC_stack_push(KLC_Stack*, const char*, const char*, size_t);
void KLC_stack_pop(KLC_Stack*);

void KLC_panic_with_error(KLC_Error* error);
KLC_Error* KLC_new_error_with_message(KLC_Stack*, const char*);
KLC_Error* KLC_errorf(size_t hint, KLC_Stack*, const char*, ...);
void KLC_delete_error(KLC_Error*);
const char* KLC_get_error_message(KLC_Error*);
void KLC_panic(const char*);

KLC_bool KLC_is(KLC_var left, KLC_var right);

void KLC_retain(KLC_Header*);
void KLC_release(KLC_Header*);
void KLC_partial_release(KLC_Header*, KLC_Header**);
void KLC_retain_var(KLC_var);
void KLC_release_var(KLC_var);
void KLC_partial_release_var(KLC_var, KLC_Header**);
void KLC_release_on_exit(KLC_Header*);
void KLC_release_var_on_exit(KLC_var);
void KLC_release_vars_queued_for_exit();

void* KLC_realloc_var_array(void* buffer, size_t old_cap, size_t new_cap);
void KLC_partial_release_var_array(
  void* buffer, size_t size, size_t cap, void* delete_queue);
void KLC_var_array_clear_range(void* buffer, size_t begin, size_t end);
KLC_var KLC_var_array_get(void* buffer, size_t i);
void KLC_var_array_set(void* buffer, size_t i, KLC_var value);

KLC_Lambda_capture* KLC_new_Lambda_capture(size_t, ...);
KLC_var KLC_Lambda_capture_get(KLC_Lambda_capture*, size_t);
void KLC_free_lambda_capture(KLC_Lambda_capture*);
KLC_Error* KLC_lambda_call(KLC_Stack*, KLC_var*, int, KLC_var*);

KLC_var KLC_var_from_ptr(KLC_Header* p);
KLC_var KLC_var_from_bool(KLC_bool b);
KLC_var KLC_var_from_int(KLC_int i);
KLC_var KLC_var_from_float(KLC_float f);
KLC_var KLC_var_from_type(KLC_Class* c);
KLC_Error* KLC_var_to_ptr(KLC_Stack*, KLC_Header** out, KLC_var, KLC_Class*);
KLC_Error* KLC_var_to_bool(KLC_Stack*, KLC_bool* out, KLC_var);
KLC_Error* KLC_var_to_int(KLC_Stack*, KLC_int* out, KLC_var);
KLC_Error* KLC_var_to_float(KLC_Stack*, KLC_float* out, KLC_var);
KLC_Error* KLC_var_to_type(KLC_Stack*, KLC_Class** out, KLC_var);

KLC_Class* KLC_get_class(KLC_var);
KLC_MethodEntry* KLC_find_instance_method(KLC_Class*, const char*);
KLC_MethodEntry* KLC_find_class_method(KLC_Class*, const char*);
KLC_bool KLC_has_method(KLC_var, const char*);
KLC_Error* KLC_call_method(KLC_Stack*, KLC_var*, const char*, int, KLC_var*);

KLC_int KLC_get_tag(KLC_var x);
KLC_int KLC_get_obj_tag();
KLC_int KLC_get_bool_tag();
KLC_int KLC_get_int_tag();
KLC_int KLC_get_float_tag();
KLC_int KLC_get_type_tag();
char const* KLC_get_type_module_name(KLC_Class* t);
char const* KLC_get_type_short_name(KLC_Class* t);

/* Useful structs and functions that are defined by builtins.k */
struct KLCCSbuiltins_DString;
struct KLCCSbuiltins_DString* KLC_str(char const *KLCLPcstr);
char const *KLC_cstr(struct KLCCSbuiltins_DString* KLCLPs);

KLC_Error* KLC_new_error_from_string(
  KLC_Stack* stack,
  struct KLCCSbuiltins_DString*);

KLC_var KLC_mklist(); /* For building lists */
void KLC_list_push(KLC_var list, KLC_var x);

/* Expected entry point */
KLC_Error *KLCFNmain_Dmain(KLC_Stack *KLC_stack, void *KLC_output_ptr);

#endif/*kcrt_h*/
