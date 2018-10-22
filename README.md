# KL
KLC = Kyumin Language Compiler

### Getting started

KLC is really a transpiler that takes k files and emits a single
C translation unit source. Then you can compile the generated C
code to run the code.

So to play with this you need:
  * python (3.6+), and
  * c89 compliant C compiler
    - basically any modern C compiler should be c89 compliant

Example usage of the compiler:

```
klc tests/sanity.k && ./a.out
```

## TL;DR

It's meant to feel like C with:
* reference counting
* classes and traits
* reflection

## Examples

### Hello world
```
// Comments look like this~
void main() {
  print('Hello world!')
}
```

### Reading from a file
```
void main() {
  final file = File("my-file-name.txt", 'r')
  final contents = file.read()
  print(contents)
}
```

### Mixins
```
// Kind of like multiple inheritance.
// However, it is not possible to inherit from
// classes, only from traits

trait A {
  var a() {
    return 'hi'
  }
}

trait B {
  String b() {
    return 'result of b'
  }
}

class C(A, B) {
}

void main() {
  final c = C()
  print(c.a())
  print(c.b())
}
```

### Fizzbuzz
```
void main() {
  int n = 50
  int i = 1

  while (i <= 50) {
    if (i % 3 == 0 and i % 5 == 0) {
      print('fizzbuzz')
    } else if (i % 3 == 0) {
      print('fizz')
    } else if (i % 5 == 0) {
      print('buzz')
    } else {
      print(i)
    }
    i = i + 1
  }
}
```

## TODO

* cleaner lambda expressions
  Lambda expressions have now been implemented,
  but due to the limitations of the current way things are done,
  lambdas have to explicitly list which variables to capture
  and their types. Ideally, the compiler should be able to
  deduce this information.
