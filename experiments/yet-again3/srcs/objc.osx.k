// Objective C bindings
#include "objc/objc-runtime.h"
#lib objc
from c import printf
typedef * id
typedef * Class(id)
typedef * Method
typedef * BOOL
typedef * SEL

extern Class objc_getClass(c_char const* name)


// Working with Classes
extern c_char const* class_getName(Class cls)
extern Class class_getSuperclass(Class cls)
extern BOOL class_isMetaClass(Class cls)
extern size_t class_getInstanceSize(Class cls)
extern Method* class_copyMethodList(Class cls, c_unsigned_int* outCount)

// Working with Selectors
extern c_char const* sel_getName(SEL sel)
extern SEL sel_registerName(c_char const* str)

// Working with Methods
extern SEL method_getName(Method m)
extern c_char const* method_getTypeEncoding(Method m)


void asdf_dump_methods(Class cls) {
  c_unsigned_int outCount
  Method* methods = class_copyMethodList(cls, &outCount)
  int n = static_cast(int, outCount)
  int i = 0
  while (i < n) {
    Method m = methods[i]
    printf(
      c"  method name = %s, type encoding = %s\n",
      sel_getName(method_getName(m)),
      method_getTypeEncoding(m),
    )
    i = i + 1
  }
}
