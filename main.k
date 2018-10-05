trait Tr {
  String f() {
    return 'From Tr.f()';
  }
}
class Foo(Tr) {
  String g(int x) {
    return str(x);
  }
}
class Bar {
  double d;

  int f() {
    return 10;
  }
}

class Baz {
  new(int a) {
    print('Baz.a');
  }
}

type type(var x);
String str(var x);
void puts(String s);

void print(var x) {
  puts(str(x));
}

int ff() {
  return 184;
}

String _f() {
  return 'result of _f';
}

void main() {
  String s = 'String s variable';
  var v = 'var v variable';
  print('Hello world!');
  print('Hello world!'.size());
  print(s);
  print(s.size());
  print(v);
  print(v.size());
  print(ff());
  print(15 + 17);
  var i = 81;
  print(i + 5);
  print(81 + 5);
  print(i - 5);
  print(81 == 81);
  print(_f());
  Bar bar = Bar();
  print(bar);
  Baz baz = Baz(15);
  print(baz);
}
