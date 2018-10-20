Style guide

## Nullable return value naming convention

The name of any function that can potentially return a null type
should end with 'OrNull'

## Naming global symbols in modules

The name of global symbols in modules should be prefixed by all caps of
the modules name.

So for instance, a symbol in the module 'sdl' in file 'sdl.k'
could be `SDLContext`.

## Lower/upper case naming convention

* Class names have the following form:
  * for builtins, they are always MixedCase
  * for package classes, they should follow PREFIXEDMixedCase
* Global variable names should
  * use ALL_CAPS
* local variables, functions and method names should
  * use camelCase
