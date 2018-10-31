class Source {
  String name
  String data

  new(String name, String data) {
    this.name = name
    this.data = data
  }

  String Repr() {
    return '<Source "' + this.name + '">'
  }
}

class Token {
  Source source
  int start
  int end
  String type
  var data

  new(Source source, int start, int end, String type, var data) {
    this.source = source
    this.start = start
    this.end = end
    this.type = type
    this.data = data
  }
}
