#include "./a.k"
#include "./b.k"

class Bar {
  double d;
  String s;

  int f() {
    return 10;
  }
}

class Baz {
  new(int a) {
    print('Baz.a');
  }
}

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
  print('Hello world!'.size);
  print(s);
  print(s.size);
  print(v);
  print(v.size);
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

  bar.d = 10.;
  print('bard.d = ' + str(bar.d));

  print(bar.s);
  bar.s = 'bar.s content';
  print(bar.s);

  {
    String vs = v.(String);
    print('vs = ' + vs);
  }

  if (1 < 10) {
    print('1 < 10 is true');
  } else if (1 == 10) {
    print('1 == 10 10 is true');
  } else {
    print('1 < 10 is false, or something');
  }
}
