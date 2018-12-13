from objc import id
from objc import Class
from objc import objc_getClass
from objc import class_getName
from objc import class_copyMethodList
from c import printf
from objc import asdf_dump_methods

void main() {
  Class cls = objc_getClass(c"NSObject")
  // id clsid = dynamic_cast(id, cls)
  printf(c"class name = %s\n", class_getName(cls))

  asdf_dump_methods(cls)
}
