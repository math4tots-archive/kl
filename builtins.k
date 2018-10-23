
#(define C_HEADERS [])
#(define C_SOURCES [
  'main.c'    // This is where the main program is
])

// Frameworks that are required for deploying to darwin targets.
#(define APPLE_FRAMEWORKS [])

// Special functions required by the language
bool _Is(var a, var b)
bool _IsNot(var a, var b)
bool _Eq(var a, var b)
bool _Ne(var a, var b)
bool _Lt(var a, var b)
bool _Le(var a, var b)
bool _Gt(var a, var b)
bool _Ge(var a, var b)

// These are explicit function names for primitive types
// These methods must also be explicitly listed in the
// _primitive_method_names dictionary above, and of course
// actually implemented in the C code.
// TODO: Implement a more elegant mechanism for enumerating
// primitive type methods.
String `null:Repr`(var x) {
  return 'null'
}
bool `null:Bool`(var x) {
  return false
}
bool `null:Eq`(var a, var b) {
  return a is b
}

String `bool:Repr`(bool b) {
  return b ? 'true' : 'false'
}
bool `bool:Bool`(bool b) {
  return b;
}
bool `bool:Eq`(bool a, bool b) {
  return a is b;
}

int `int:Add`(int a, int b)
int `int:Sub`(int a, int b)
int `int:Mul`(int a, int b)
int `int:Div`(int a, int b)
int `int:Mod`(int a, int b)
bool `int:Eq`(int a, int b)
bool `int:Lt`(int a, int b)
String `int:Repr`(int a)
bool `int:Bool`(int x) {
  return x != 0
}
int `int:HashCode`(int x) {
  return x
}

double `double:Add`(double a, double b)
double `double:Sub`(double a, double b)
double `double:Mul`(double a, double b)
double `double:Div`(double a, double b)
bool `double:Eq`(double a, double b)
bool `double:Lt`(double a, double b)
String `double:Repr`(double a)
bool `double:Bool`(double x) {
  return x != 0.0
}

String `function:GETname`(function f)
String `function:Repr`(function f) {
  return '<function ' + f.name + '>'
}
bool `function:Bool`(function f) {
  return true
}

String `type:GETname`(type t)
String `type:Repr`(type t) {
  return '<type ' + t.name + '>'
}
bool `type:Bool`(type t) {
  return true
}
bool `type:Eq`(type a, type b)

trait Object() {
  bool Eq(var other) {
    return this is other
  }
  int HashCode() {
    // TODO: Change to using pointer value
    // by default
    return 0
  }
  String Repr() {
    type t = type(this)
    return '<' + t.name + ' instance>'
  }
  String Str() {
    return this.Repr().(String)
  }
  bool Bool() {
    return true
  }
}

extern class WeakReference {
  new(var obj)
  var getNullable()

  bool Bool() {
    return this.getNullable() is not null
  }

  var get() {
    var obj = this.getNullable()
    if (obj is null) {
      panic("Get null weak reference")
    }
    return obj
  }

  int GETrefcnt()
}

extern class String {
  int GETbytesize()
  String Add(String b)
  int GETsize()
  String Str()
  String Repr() {
    return '"' + this.escape() + '"'
  }
  String escape()
  bool Eq(String other)
  bool Lt(String other)
  bool Bool() {
    return bool(this.size)
  }

  String join(List args) {
    final sb = StringBuilder()
    for (int i = 0; i < args.size; i++) {
      if (i > 0) {
        sb.add(this)
      }
      sb.add(args[i])
    }
    return sb.build()
  }

  String Slice(int start, int end)

  String SliceAll() {
    // Since Strings are immutable, it's safe to
    // return this
    return this
  }

  String SliceLeft(int start) {
    return this.Slice(start, this.size)
  }

  String SliceRight(int end) {
    return this.Slice(0, end)
  }

  String GetItem(int i) {
    if (i < 0) {
      i = i + this.size
    }
    return this.Slice(i, i + 1)
  }
}

extern class StringBuilder {
  new()
  void addstr(String s)

  // NOTE: The 'build' method empties out the StringBuilder
  String build()

  void add(var x) {
    this.addstr(str(x))
  }
}

extern class File {
  new(String path, String mode)
  void close()
  void write(String s)
  String read()
  void print(var x) {
    this.write(str(x))
    this.write('\n')
  }
}

extern class Closure {
  new(List captures, function f)
  extern Call
}

extern File STDIN
extern File STDOUT
extern File STDERR

// argc and argv passed to main
// made available as a List
extern List ARGS

void panic(String message)
bool bool(var x)
type type(var x)
void assert(var x)

void print(var x) {
  STDOUT.print(x)
}

String str(var x) {
  return x.Str().(String)
}

String repr(var x) {
  return x.Repr().(String)
}

trait Iterable {
  void forEach(var f) {
    final i = this.Iterator()
    while (i.HasNext()) {
      f(i.Next())
    }
  }
}

List ListFromIterable(var iterable) {
  final ret = []
  for x in iterable {
    ret.push(x)
  }
  return ret
}

extern class List(Iterable) {
  int GETsize()
  var GetItem(int i)
  var SetItem(int i, var x)
  void push(var x)

  String Repr() {
    StringBuilder sb = StringBuilder()
    sb.addstr('[')
    int size = this.size
    int i = 0
    while (i < size) {
      if (i != 0) {
        sb.addstr(', ')
      }
      sb.addstr(repr(this[i]))
      i = i + 1
    }
    sb.addstr(']')
    return sb.build()
  }

  bool Eq(List other) {
    int size = this.size
    if (size != other.size) {
      return false
    }
    for (int i = 0; i < size; i++) {
      if (this[i] != other[i]) {
        return false
      }
    }
    return true
  }

  List Mul(int rep) {
    final ret = []
    for (int i = 0; i < rep; i++) {
      for item in this {
        ret.push(item)
      }
    }
    return ret
  }

  bool Contains(var x) {
    int size = this.size
    for (int i = 0; i < size; i++) {
      if (this[i] == x) {
        return true
      }
    }
    return false
  }

  ListIterator Iterator() {
    return ListIterator(this)
  }

  List Slice(int a, int b) {
    if (a < 0) {
      a = a + this.size
    }
    if (b < 0) {
      b = b + this.size
    }
    final ret = []
    for (int i = a; i < b; i++) {
      ret.push(this[i])
    }
    return ret
  }

  List SliceRight(int b) {
    return this.Slice(0, b)
  }

  List SliceLeft(int a) {
    return this.Slice(a, this.size)
  }

  List SliceAll() {
    return this.Slice(0, this.size)
  }

  List Add(List other) {
    final ret = []
    for x in this {
      ret.push(x)
    }
    for x in other {
      ret.push(x)
    }
    return ret
  }

  List map(var f) {
    final ret = []
    for x in this {
      ret.push(f(x))
    }
    return ret
  }
}

class ListIterator {
  List list
  int i

  new(List list) {
    this.list = list
    this.i = 0
  }

  bool HasNext() {
    return this.i < this.list.size
  }

  var Next() {
    var value = this.list[this.i]
    this.i = this.i + 1
    return value
  }
}

extern class Try {
  new(bool success, var value)
  bool Bool()
  var getRawValue()

  var orElse(var callback) {
    var value = this.getRawValue()
    if (!this) {
      return callback(value)
    }
    return value
  }

  var orDie() {
    var value = this.getRawValue()
    if (!this) {
      panic(str(value))
    }
    return value
  }

  String Repr() {
    final value = repr(this.getRawValue())
    return (
      this ?
        ('<Try(Success) ' + value + '>') :
        ('<Try(Failue) ' + value + '>')
    )
  }
}

class Map {
  // Dead simple hash table implementation
  //   contains a list of buckets,
  //   each bucket is a list of entries,
  //   each entry is a triple of [hash, key, value]
  List _buckets
  int _size

  new() {
    // Buckets count should never be zero
    this._buckets = [null]
    this._size = 0
  }

  bool Eq(Map other) {
    if (this.size != other.size) {
      return false
    }
    for key in this {
      if (key not in other or this[key] != other[key]) {
        return false
      }
    }
    return true
  }

  void _resizeBuckets(int newSize) {
    final oldBuckets = this._buckets
    this._buckets = [null] * newSize
    this._size = 0
    for bucket in oldBuckets {
      if (bucket is not null) {
        for entry in bucket.(List) {
          int hash = entry[0].(int)
          var key = entry[1]
          var value = entry[2]
          this._insertWithHash(hash, key, value)
        }
      }
    }
  }

  List _bucketFor(int hash) {
    final index = hash % this._buckets.size
    if (this._buckets[index] is null) {
      this._buckets[index] = []
    }
    return this._buckets[index].(List)
  }

  void _maybeResize() {
    if (this._buckets.size <= 2 * this._size) {
      this._resizeBuckets(2 * (this._buckets.size + this._size + 8))
    }
  }

  void _insertWithHash(int hash, var key, var value) {
    final bucket = this._bucketFor(hash)

    // First check if this value exists. If so, just update
    // the existing value.
    for entry in bucket {
      if (entry[0].(int) == hash and entry[1] == key) {
        entry[2] = value
        return
      }
    }

    // Otherwise, create a new entry
    bucket.push([hash, key, value])
    this._size = this._size + 1
    this._maybeResize()
  }

  void _insert(var key, var value) {
    this._insertWithHash(key.HashCode().(int), key, value)
  }

  List _findWithHash(int hash, var key) {
    final bucket = this._bucketFor(hash)
    for entry in bucket {
      if (entry[0].(int) == hash and entry[1] == key) {
        return entry.(List)
      }
    }
    return null(List)
  }

  List _findKey(var key) {
    return this._findWithHash(key.HashCode().(int), key)
  }

  bool remove(var key) {
    int hash = key.HashCode().(int)
    int index = hash % this._buckets.size
    final bucket = this._buckets[index].(List)
    for (int i = 0; i < bucket.size; i++) {
      final entry = bucket[i].(List)
      if (entry[0].(int) == hash and entry[1] == key) {
        if (bucket.size == 1) {
          this._buckets[index] = null
        } else {
          this._buckets[index] = bucket[:i] + bucket[i + 1:]
        }
        this._size = this._size - 1
        return true
      }
    }
    return false
  }

  bool Contains(var key) {
    return this._findKey(key) is not null
  }

  var GetItem(var key) {
    final entry = this._findKey(key)
    if (entry is null) {
      panic('Key ' + str(key) + ' not found')
    }
    return entry[2]
  }

  var SetItem(var key, var value) {
    this._insert(key, value)
    return value
  }

  int GETsize() {
    return this._size
  }

  MapIterator Iterator() {
    return MapIterator(this)
  }

  String Repr() {
    return '{' + ', '.join(
      ListFromIterable(this).map(def String[Map this](var key) {
        return repr(key) + ': ' + repr(this[key])
      })
    ) + '}'
  }
}

class MapIterator {
  List _buckets
  int _i
  int _j

  new(Map m) {
    this._buckets = m._buckets
    this._i = 0  // index to pick which bucket,
    this._j = 0  // index to pick which entry in the bucket
    this._skipEmpty()
  }

  void _skipEmpty() {
    List bs = this._buckets
    int nb = bs.size
    int i = this._i
    int j = this._j
    if (i < nb and bs[i] is not null and j >= bs[i].(List).size) {
      j = 0
      i++
    }
    if (i < nb and bs[i] is null) {
      i++
      while (i < nb and bs[i] is null) {
        i++
      }
      j = 0
    }
    this._i = i
    this._j = j
  }

  bool HasNext() {
    return this._i < this._buckets.size
  }

  var Next() {
    if (!this.HasNext()) {
      panic('MapIterator:Next no more')
    }
    var next = this._buckets[this._i].(List)[this._j].(List)[1]
    this._j = this._j + 1
    this._skipEmpty()
    return next
  }
}

Map MapFromPairs(List pairs) {
  final ret = Map()
  for pairvar in pairs {
    final pair = pairvar.(List)
    ret[pair[0]] = pair[1]
  }
  return ret
}

class Set {
  Map _map

  new() {
    this._map = Map()
  }

  bool Eq(Set other) {
    return this._map == other._map
  }

  int GETsize() {
    return this._map.size
  }

  void add(var key) {
    this._map[key] = 1
  }

  bool Contains(var key) {
    return key in this._map
  }

  bool remove(var key) {
    return this._map.remove(key)
  }

  MapIterator Iterator() {
    return this._map.Iterator()
  }

  String Repr() {
    return '{' + ', '.join(ListFromIterable(this).map(repr)) + '}'
  }
}

Set SetFromList(List list) {
  final ret = Set()
  for x in list {
    ret.add(x)
  }
  return ret
}
