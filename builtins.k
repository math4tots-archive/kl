#(global C_HEADERS [])
#(global C_SOURCES [
  'generated_main.c'    // This is where the main program is
])

// Frameworks that are required for deploying to darwin targets.
#(global APPLE_FRAMEWORKS [])

#(global assert (lambda (condition message)
  (if
      condition 1
      (error 'assertion failed')
    )
))

#(global not (lambda (condition)
  (if condition 0 1)
))

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
int `null:HashCode`(var x) {
  return 647098
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
bool `bool:Lt`(bool a, bool b) {
  return !a and b
}
int `bool:HashCode`(bool b) {
  return b ? 1231 : 1237
}

int `int:Add`(int a, int b)
int `int:Sub`(int a, int b)
int `int:Mul`(int a, int b)
int `int:Div`(int a, int b)
int `int:Mod`(int a, int b)
bool `int:Eq`(int a, int b)
bool `int:Lt`(int a, int b)
int `int:Xor`(int a, int b)
int `int:And`(int a, int b)
int `int:Or`(int a, int b)
int `int:Lshift`(int a, int b)
int `int:Rshift`(int a, int b)
int `int:Invert`(int a)
String `int:Repr`(int a)
bool `int:Bool`(int x) {
  return x != 0
}
int `int:HashCode`(int x) {
  return x
}
int `int:Pow`(int a, int b) {
  // TODO: Consider faster exponentiation algorithm
  int r = 1
  for (int i = 0; i < b; i++) {
    r = r * a
  }
  return r
}

double `double:Add`(double a, double b)
double `double:Sub`(double a, double b)
double `double:Mul`(double a, double b)
double `double:Div`(double a, double b)
bool `double:Eq`(double a, double b)
bool `double:Lt`(double a, double b)
int `double:HashCode`(double a)
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
bool `function:Eq`(function a, function b) {
  return a is b
}
int `function:HashCode`(function a) {
  return _IdentityHash(a)
}

String `type:GETname`(type t)
String `type:Repr`(type t) {
  return '<type ' + t.name + '>'
}
bool `type:Bool`(type t) {
  return true
}
bool `type:Eq`(type a, type b)
int `type:HashCode`(type a) {
  return _IdentityHash(a)
}

int _IdentityHash(var x)

trait Object() {
  bool Eq(var other) {
    return this is other
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

int ord(String c)

extern class String {
  int GETbytesize()
  String Add(String b)
  int GETsize()
  int HashCode()

  Buffer encodeUtf8()

  Buffer encode(String encoding) {
    if (encoding == 'utf-8' or encoding == 'utf8') {
      return this.encodeUtf8()
    }
    panic('Unknown encoding ' + encoding)
    return null(Buffer)
  }

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

  String Mul(int rep) {
    final sb = StringBuilder()
    for (int i = 0; i < rep; i++) {
      sb.add(this)
    }
    return sb.build()
  }

  bool startsWith(String prefix) {
    return this.startsWithAt(prefix, 0)
  }

  bool startsWithAt(String prefix, int pos) {
    if (pos + prefix.size > this.size) {
      return false
    }
    for (int i = 0; i < prefix.size; i++) {
      if (this[pos + i] != prefix[i]) {
        return false
      }
    }
    return true
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

extern int CHAR_BIT
extern int INT_MAX
extern int INT_MIN

extern class Buffer {
  // Assumes CHAR_BITS = 8
  // Also, everything is cast to KLC_int.
  // get4/set4 may not behave as expected if KLC_int
  // is only 32 bits, since KLC_int is signed.
  // TODO: Fix this
  new(int size)
  int get1(int i)
  void set1(int i, int v)
  int GETsize()
  bool Eq(Buffer other)
  String decodeUtf8()
  void resize(int newSize)

  String decode(String encoding) {
    if (encoding == 'utf-8' or encoding == 'utf8') {
      return this.decodeUtf8()
    }
    panic('Unknown encoding ' + encoding)
    return null(String)
  }

  int get2(int i) {
    return this.get1(i) + (2 ** CHAR_BIT) * this.get1(i + 1)
  }

  int get4(int i) {
    return this.get2(i) + (2 ** (2 * CHAR_BIT)) * this.get2(i + 2)
  }

  void set2(int i, int v) {
    this.set1(i, v % (2 ** CHAR_BIT))
    this.set1(i + 1, v / (2 ** CHAR_BIT))
  }

  void set4(int i, int v) {
    this.set2(i, v % (2 ** (2 * CHAR_BIT)))
    this.set2(i + 2, v / (2 ** (2 * CHAR_BIT)))
  }

  String dumpstr() {
    final sb = StringBuilder()
    sb.add('<Buffer ')
    final size = this.size
    for (int i = 0; i < size; i++) {
      if (i != 0) {
        sb.add(', ')
      }
      sb.add(this.get1(i))
    }
    sb.add('>')
    return sb.build()
  }

  BufferView view(int start, int end) {
    return BufferView(this, start, end)
  }

  String Repr() {
    return 'Buffer(' + str(this.size) + ')'
  }
}

class BufferView {
  Buffer buffer
  int start
  int end

  new(Buffer buffer, int start, int end) {
    this.buffer = buffer
    this.start = start
    this.end = end
  }

  int GETsize() {
    return this.end - this.start
  }

  void _checkBounds(int i) {
    if (i < 0 and i >= this.size) {
      panic('Index out of bounds')
    }
  }

  int get1(int i) {
    this._checkBounds(i)
    return this.buffer.get1(this.start + i)
  }

  int get2(int i) {
    this._checkBounds(i)
    this._checkBounds(i + 1)
    return this.buffer.get2(this.start + i)
  }

  int get4(int i) {
    this._checkBounds(i)
    this._checkBounds(i + 3)
    return this.buffer.get4(this.start + i)
  }
}

Buffer BufferFromByteList(List bytes) {
  int size = bytes.size
  Buffer ret = Buffer(size)
  for (int i = 0; i < size; i++) {
    ret.set1(i, bytes[i].(int))
  }
  return ret
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
  void close()
  void write(String s)
  String read()
  void print(var x) {
    this.write(str(x))
    this.write('\n')
  }
  File Enter() {
    return this
  }
  void Exit() {
    this.close()
  }
}

File open(String path, String mode)

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
int errno()
String strerror(int e)

int hash(var x) {
  return x.HashCode().(int)
}

void print(var x) {
  STDOUT.print(x)
}

String str(var x) {
  return x.Str().(String)
}

String repr(var x) {
  return x.Repr().(String)
}

extern class `%With` {
  // Helper class for auto-closing in with statements
  new(var context)
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
  new(int initSize)
  int GETsize()
  var GetItem(int i)
  var SetItem(int i, var x)
  void push(var x)
  void resize(int newSize)
  var pop()
  void sort()

  bool Bool() {
    return bool(this.size)
  }

  void removeSlice(int a, int b) {
    for (int i = a; i < b; i++) {
      this[i] = this[i + b - a]
    }
    this.resize(this.size - b + a)
  }

  void removeAt(int index) {
    this.removeSlice(index, index + 1)
  }

  void clear() {
    while (this) {
      this.pop()
    }
  }

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
    final ret = List(rep * this.size)
    int p = 0
    for (int i = 0; i < rep; i++) {
      for item in this {
        ret[p] = item
        p++;
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

List sorted(var iterable) {
  final xs = ListFromIterable(iterable)
  xs.sort()
  return xs
}

List reversed(var iterable) {
  final xs = (
    type(iterable) == List ? iterable.(List) : ListFromIterable(iterable)
  )
  final ret = []
  for (int i = xs.size - 1; i >= 0; i--) {
    ret.push(xs[i])
  }
  return ret
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

Try `%Success`(var value) {
  return Try(true, value)
}

Try `%Failure`(var value) {
  return Try(false, value)
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
    this._buckets = List(16)
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
    this._buckets = List(newSize)
    this._size = 0
    for bucket in oldBuckets {
      if (bucket is not null) {
        for entry in bucket.(List) {
          int h = entry[0].(int)
          var key = entry[1]
          var value = entry[2]
          this._insertWithHash(h, key, value)
        }
      }
    }
  }

  void _maybeResize() {
    if (this._buckets.size <= 2 * this._size) {
      this._resizeBuckets(4 * (this._buckets.size + 8))
    }
  }

  void _insertWithHash(int hash, var key, var value) {
    final index = hash % this._buckets.size
    final bucketvar = this._buckets[index]

    // First check if this value exists. If so, just update
    // the existing value.
    if (bucketvar is not null) {
      final bucket = bucketvar.(List)
      for entry in bucket {
        if (entry[0].(int) == hash and entry[1] == key) {
          entry[2] = value
          return
        }
      }
    }

    // Otherwise, create a new entry
    if (bucketvar is null) {
      this._buckets[index] = [[hash, key, value]]
    } else {
      bucketvar.(List).push([hash, key, value])
    }
    this._size = this._size + 1
    this._maybeResize()
  }

  void _insert(var key, var value) {
    this._insertWithHash(hash(key), key, value)
  }

  List _findWithHash(int hash, var key) {
    final index = hash % this._buckets.size
    final bucketvar = this._buckets[index]
    if (bucketvar is not null) {
      final bucket = bucketvar.(List)
      for entry in bucket {
        if (entry[0].(int) == hash and entry[1] == key) {
          return entry.(List)
        }
      }
    }
    return null(List)
  }

  List _findKey(var key) {
    return this._findWithHash(hash(key), key)
  }

  bool remove(var key) {
    int h = hash(key)
    int index = h % this._buckets.size
    final bucket = this._buckets[index].(List)
    for (int i = 0; i < bucket.size; i++) {
      final entry = bucket[i].(List)
      if (entry[0].(int) == h and entry[1] == key) {
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
