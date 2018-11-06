class Source {
  String name
  String path
  String data

  new(String name, String path, String data) {
    this.name = name
    this.path = path
    this.data = data
  }
}

class Token {
  Source source
  int i
  String type
  var data

  new(Source source, int i, String type, var data) {
    this.source = source
    this.i = i
    this.type = type
    this.data = data
  }

  String Repr() {
    return 'Token(' + repr(this.type) + ', ' + repr(this.data) + ')'
  }

  int GETlineno() {
    int ln = 1
    final x = this.i
    final s = this.source.data
    for (int i = 0; i < x; i++) {
      if (s[i] == '\n') {
        ln++
      }
    }
    return ln
  }

  int GETcolno() {
    int cn = 1
    final s = this.source.data
    int i = this.i
    final size = s.size
    while (i > 0 and s[i - 1] != '\n') {
      i--
      cn++
    }
    return cn
  }

  String GETline() {
    final s = this.source.data
    final size = s.size
    int start = this.i
    while (start > 0 and s[start - 1] != '\n') {
      start--
    }
    int end = this.i
    while (end < size and s[end] != '\n') {
      end++
    }
    return s[start:end]
  }
}

class Error {
  List tokens
  String message

  new(List tokens, String message) {
    this.tokens = tokens
    this.message = message
  }

  String Repr() {
    return 'klc.ast.Error(' + this.message + ')'
  }

  String Str() {
    final sb = StringBuilder()
    sb.add('KLC error: ')
    sb.add(this.message)
    sb.add('\n')
    for Token token in this.tokens {
      sb.add('On line ' + str(token.lineno) + '\n')
      sb.add(token.line + '\n')
      sb.add(' ' * (token.colno - 1) + '*\n')
    }
    return sb.build()
  }
}

Map TranslationUnit(Token token, String name, List imports, List definitions) {
  return {
    '@type': 'TranslationUnit',
    '@token': token,
    'name': name,
    'imports': imports,
    'definitions': definitions,
  }
}

Map Import(Token token, String name, String alias) {
  return {
    '@type': 'Import',
    '@token': token,
    'name': name,
    'alias': alias,
  }
}

Map FunctionDefinition(
    Token token,
    String returnType,
    String name,
    List parameters,
    Map body) {
  return {
    '@type': 'FunctionDefinition',
    '@token': token,
    'returnType': returnType,
    'name': name,
    'parameters': parameters,
    'body': body,
  }
}

Map Parameter(Token token, String type, String name) {
  return {
    '@type': 'Parameter',
    '@token': token,
    'type': type,
    'name': name,
  }
}

Map Block(Token token, List statements) {
  return {
    '@type': 'Block',
    '@token': token,
    'statements': statements,
  }
}
