import klc.ast


Set keywords = {
  'is', 'not', 'null', 'true', 'false', 'new', 'and', 'or', 'in',
  'inline', 'extern', 'class', 'trait', 'final', 'def', 'auto',
  'for', 'if', 'else', 'while', 'break', 'continue', 'return',
  'with', 'from', 'import', 'as',
}

List symbols = reversed(sorted([
  '\n',
  '||', '&&', '|', '&', '<<', '>>', '~',
  ';', '#', '?', ':', '!', '++', '--', '**',
  '.', ',', '!', '@', '^', '&', '+', '-', '/', '%', '*', '.', '=', '==', '<',
  '>', '<=', '>=', '!=', '(', ')', '{', '}', '[', ']',
]))

Map escapeMap = {
  'n': '\n',
  't': '\t',
  'r': '\r',
  '\\': '\\',
  '"': '"',
  "'": "'",
}

bool isdigit(String c) {
  final oc = ord(c)
  return oc >= ord('0') and oc <= ord('9')
}

bool iswordch(String c) {
  final oc = ord(c)
  return (
    c == '_' or
    isdigit(c) or
    oc >= ord('a') and oc <= ord('z') or
    oc >= ord('A') and oc <= ord('Z')
  )
}

bool isspace(String c) {
  return c == ' ' or c == '\r' or c == '\n'
}

Try tryLex(ast.Source source) {
  return Lexer(source).tryLex()
}

List lex(ast.Source source) {
  return Lexer(source).lex()
}

class Lexer {
  ast.Source source
  String s
  int i

  new(ast.Source source) {
    this.source = source
    this.s = source.data
    this.i = 0
  }

  bool startsWith(String prefix) {
    return this.s.startsWithAt(prefix, this.i)
  }

  bool more() {
    return this.i < this.s.size
  }

  String chAt(int offset) {
    return this.s[this.i + offset]
  }

  String ch() {
    return this.s[this.i]
  }

  void incr(int x) {
    this.i = this.i + x
  }

  void skipSpacesAndComments() {
    bool again = true
    while (this.more() and again) {
      again = false
      final ch = this.ch()
      if (ch == ' ' or ch == '\r' or ch == '\t') {
        this.incr(1)
        again = true
      } else if (this.startsWith('//')) {
        this.incr(2)
        while (this.more() and this.ch() != '\n') {
          this.incr(1)
        }
        again = true
      }
    }
  }

  ast.Token mktok(int i, String type, var data) {
    return ast.Token(this.source, i, type, data)
  }

  ast.Token next() {
    return this.tryNext().orDie().(ast.Token)
  }

  bool charAt(int offset, var condition) {
    return this.i + offset < this.s.size and condition(this.s[this.i + offset])
  }

  bool charAtIs(int offset, String ch) {
    return this.charAt(offset, def bool[String ch](String c) {
      return ch == c
    })
  }

  bool charIs(String ch) {
    return this.charAtIs(0, ch)
  }

  bool char(var condition) {
    return this.charAt(0, condition)
  }

  String sliceFrom(int start) {
    return this.s[start:this.i]
  }

  List lex() {
    return this.tryLex().orDie().(List)
  }

  Try tryLex() {
    List ret = []
    bool again = true
    while (again) {
      final t = this.tryNext()
      if (t) {
        final token = t.orDie().(ast.Token)
        if (token.type == 'EOF') {
          again = false
        }
        ret.push(token)
      } else {
        return t
      }
    }
    return Try(true, ret)
  }

  Try tryNext() {
    this.skipSpacesAndComments()
    if (!this.more()) {
      return Try(true, this.mktok(this.i, 'EOF', null))
    }
    final start = this.i
    // raw string literal
    if (this.startsWith('r"') or this.startsWith("r'")) {
      this.incr(1)  // for 'r'
      final qlen = this.startsWith(this.ch() * 3) ? 3 : 1
      final quote = this.s[this.i: this.i + qlen]
      this.incr(qlen)

      while (this.more() and !this.startsWith(quote)) {
        this.incr(1)
      }
      if (!this.more()) {
        return Try(false, ast.Error(
          [this.mktok(start, 'ERR', null)],
          'Unterminated raw string literal'
        ))
      }
      this.incr(qlen)

      final data = this.s[start + 1 + qlen: this.i - qlen]

      return Try(true, this.mktok(start, 'STRING', data))
    }
    // normal string literal
    if (this.charIs('"') or this.charIs("'")) {
      final qlen = this.startsWith(this.ch() * 3) ? 3 : 1
      final quote = this.s[this.i: this.i + qlen]
      final sb = StringBuilder()
      this.incr(qlen)

      while (this.more() and !this.startsWith(quote)) {
        final ch = this.ch()
        if (ch == '\\') {
          this.incr(1)
          if (!this.more()) {
            return Try(false, ast.Error(
              [this.mktok(this.i, 'ERR', null)],
              'Expected string literal escape'
            ))
          }
          final c = this.ch()
          if (c in escapeMap) {
            sb.add(escapeMap[c])
            this.incr(1)
          } else {
            return Try(false, ast.Error(
              [this.mktok(this.i, 'ERR', null)],
              'Invalid string literal escape'
            ))
          }
        } else {
          sb.add(ch)
          this.incr(1)
        }
      }
      if (!this.more()) {
        return Try(false, ast.Error(
          [this.mktok(start, 'ERR', null)],
          'Unterminated string literal'
        ))
      }
      this.incr(qlen)

      return Try(true, this.mktok(start, 'STRING', sb.build()))
    }
    if (this.char(isdigit) or
        this.charIs('.') and this.charAt(1, isdigit)) {  // number
      while (this.char(isdigit)) {
        this.incr(1)
      }
      if (this.charIs('.')) {
        this.incr(1)
        while (this.char(isdigit)) {
          this.incr(1)
        }
        return Try(true, this.mktok(start, 'FLOAT', this.sliceFrom(start)))
      }
      return Try(true, this.mktok(start, 'INT', this.sliceFrom(start)))
    }
    if (this.charIs('`')) {  // escaped identifier
      this.incr(1)
      while (this.more() and !this.charIs('`')) {
        this.incr(1)
      }
      if (!this.charIs('`')) {
        return Try(false, ast.Error(
          [this.mktok(start, 'ERR', null)],
          'Unterminated escaped identifier'))
      }
      this.incr(1)
      return Try(true, this.mktok(start, 'NAME', this.s[start + 1: this.i - 1]))
    }
    if (this.char(iswordch)) {  // Normal identifier
      while (this.char(iswordch)) {
        this.incr(1)
      }
      final name = this.sliceFrom(start)
      final type = name in keywords ? name : 'NAME'
      return Try(true, this.mktok(start, type, name))
    }
    for String symbol in symbols {  // symbol
      if (this.startsWith(symbol)) {
        this.incr(symbol.size)
        return Try(true, this.mktok(start, symbol, symbol))
      }
    }
    while (this.more() and this.char(isspace)) {
      this.incr(1)
    }
    return Try(false, ast.Error(
      [this.mktok(start, 'ERR', null)],
      'Unrecognized token: ' + this.sliceFrom(start)))
  }
}

