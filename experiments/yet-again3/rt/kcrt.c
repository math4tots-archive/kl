#include "kcrt.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct KLC_StackEntry KLC_StackEntry;

struct KLC_Error {
  char* message;
  size_t stack_entry_count;
  KLC_StackEntry* stack;
};

struct KLC_StackEntry {
  const char* file_name;
  const char* function_name;
  size_t line_number;
};

struct KLC_Stack {
  size_t size, cap;
  KLC_StackEntry* buffer;
};

char* KLC_CopyString(const char* s) {
  char* ret = (char*) malloc(sizeof(char) * (strlen(s) + 1));
  strcpy(ret, s);
  return ret;
}

KLC_Stack* KLC_new_stack() {
  KLC_Stack* stack = (KLC_Stack*) malloc(sizeof(KLC_Stack));
  stack->size = stack->cap = 0;
  stack->buffer = NULL;
  return stack;
}

void KLC_delete_stack(KLC_Stack* stack) {
  free(stack->buffer);
  free(stack);
}

void KLC_stack_push(
    KLC_Stack* stack,
    const char* file_name,
    const char* function_name,
    size_t ln) {
  if (stack->size == stack->cap) {
    stack->cap = stack->cap == 0 ? 32 : (stack->cap * 2);
    stack->buffer = (KLC_StackEntry*) realloc(
      stack->buffer, sizeof(KLC_StackEntry) * stack->cap);
  }
  stack->size++;
  stack->buffer[stack->size - 1].file_name = file_name;
  stack->buffer[stack->size - 1].function_name = function_name;
  stack->buffer[stack->size - 1].line_number = ln;
}

void KLC_stack_pop(KLC_Stack* stack) {
  stack->size--;
}

void KLC_panic_with_error(KLC_Error* error) {
  size_t i;
  fprintf(stderr, "ERROR: %s\n", error->message);
  for (i = 0; i < error->stack_entry_count; i++) {
    fprintf(
      stderr,
      "  File \"%s\", line %lu, in %s\n",
      error->stack[i].file_name,
      (unsigned long) error->stack[i].line_number,
      error->stack[i].function_name
    );
  }
  KLC_delete_error(error);
  exit(1);
}

KLC_Error* KLC_new_error_with_message(KLC_Stack* stack, const char* msg) {
  KLC_Error* error = malloc(sizeof(KLC_Error));
  size_t stack_entry_count = stack->size;
  size_t nbytes = sizeof(KLC_StackEntry) * stack_entry_count;
  error->message = KLC_CopyString(msg);
  error->stack_entry_count = stack_entry_count;
  error->stack = (KLC_StackEntry*) malloc(nbytes);
  memcpy(error->stack, stack->buffer, nbytes);
  return error;
}

void KLC_delete_error(KLC_Error* error) {
  free(error->message);
  free(error->stack);
  free(error);
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

void* KLC_realloc_var_array(void* buffer, size_t old_cap, size_t new_cap) {
  KLC_var KLC_null = {0};
  size_t i;
  KLC_var* arr = (KLC_var*) buffer;

  if (old_cap > new_cap) {
    KLC_var_array_clear_range(buffer, new_cap, old_cap);
  }

  arr = (KLC_var*) realloc(buffer, sizeof(KLC_var) * new_cap);

  if (old_cap < new_cap) {
    for (i = old_cap; i < new_cap; i++) {
      arr[i] = KLC_null;
    }
  }

  return arr;
}

void KLC_partial_release_var_array(
    void* buffer, size_t size, size_t cap, void* delete_queue) {
  KLC_var* arr = (KLC_var*) buffer;
  size_t i;
  for (i = 0; i < size; i++) {
    KLC_partial_release_var(arr[i], delete_queue);
  }
  free(buffer);
}

void KLC_var_array_clear_range(void* buffer, size_t begin, size_t end) {
  KLC_var KLC_null = {0};
  KLC_var* arr = (KLC_var*) buffer;
  size_t i;
  for (i = begin; i < end; i++) {
    KLC_release_var(arr[i]);
    arr[i] = KLC_null;
  }
}

KLC_var KLC_var_array_get(void* buffer, size_t i) {
  KLC_var* arr = (KLC_var*) buffer;
  KLC_retain_var(arr[i]);
  return arr[i];
}

void KLC_var_array_set(void* buffer, size_t i, KLC_var value) {
  KLC_var* arr = (KLC_var*) buffer;
  KLC_retain_var(value);
  KLC_release_var(arr[i]);
  arr[i] = value;
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

KLC_Error* KLC_var_to_ptr(
    KLC_Stack* stack, KLC_Header** out, KLC_var v, KLC_Class* cls) {
  if (v.tag != KLC_TAG_POINTER) {
    return KLC_new_error_with_message(stack, "Expected class type");
  }
  if (v.u.p->cls != cls) {
    return KLC_new_error_with_message(
        stack, "Tried to cast to incorrect class type");
  }
  *out = v.u.p;
  return NULL;
}

KLC_Error* KLC_var_to_int(KLC_Stack* stack, KLC_int* out, KLC_var v) {
  if (v.tag != KLC_TAG_INT) {
    return KLC_new_error_with_message(stack, "Expected integral type");
  }
  *out = v.u.i;
  return NULL;
}

KLC_Error* KLC_var_to_float(KLC_Stack* stack, KLC_float* out, KLC_var v) {
  if (v.tag != KLC_TAG_FLOAT) {
    return KLC_new_error_with_message(stack, "Expected float type");
  }
  *out = v.u.f;
  return NULL;
}

KLC_Class* KLC_get_class(KLC_var v) {
  switch (v.tag) {
    case KLC_TAG_POINTER:
      if (v.u.p) {
        return v.u.p->cls;
      }
  }
  return NULL;
}

KLC_MethodEntry* KLC_find_method(KLC_Class* cls, const char* name) {
  size_t lower = 0;
  size_t upper = cls->number_of_methods;
  if (cls->number_of_methods == 0) {
    return NULL;
  }
  for (;;) {
    size_t mid = (lower + upper) / 2;
    int cmp = strcmp(cls->methods[mid].name, name);
    if (cmp == 0) {
      return cls->methods + mid;
    } else if (lower == mid) {
      break;
    } else if (cmp < 0) {
      upper = mid;
    } else {
      lower = mid;
    }
  }
  return NULL;
}

KLC_Error* KLC_call_method(
    KLC_Stack* stack, KLC_var* out, const char* name, int argc, KLC_var* argv) {
  KLC_Class* cls;
  KLC_MethodEntry* method_entry;
  if (argc < 1) {
    return KLC_new_error_with_message(stack, "FUBAR: KLC_call_method with argc < 1");
  }
  /* TODO: Method call for primitive types */
  cls = KLC_get_class(argv[0]);
  if (!cls) {
    int* p = NULL;
    int i = *p;
    printf("i = %d\n", i);
    return KLC_new_error_with_message(stack, "FUBAR: Could not get class for method call");
  }
  method_entry = KLC_find_method(cls, name);
  if (!method_entry) {
    return KLC_new_error_with_message(stack, "Method not found");
  }
  return method_entry->method(stack, out, argc, argv);
}
