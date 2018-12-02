# Kyumin C

Like C but with 3 features:
  * namespaces
  * reference counting
  * exceptions
  * (Built-in C-interop)

This is all I'm shooting for right now.

# TODO

* Implement method calls
* Allow manually releasing from delete hooks.
  Syntax idea: use 'delete <expr>' form to
  manually 'partial_release' a var or object.
  This feature is necessary to implement a 'List' type.
