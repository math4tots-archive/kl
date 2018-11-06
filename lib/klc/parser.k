import klc.ast
import klc.lexer


void die(List tokens, String message) {
  Try(false, ast.Error(tokens, message)).orDie()
}

Map parse(ast.Source source) {
  return Parser(source, lexer.lex(source)).parseTranslationUnit()
}

Map parseString(String data) {
  return parse(ast.Source('#string', null(String), data))
}

class Parser {
  ast.Source source
  List tokens
  int i
  List skipNewlineStack

  new(ast.Source source, List tokens) {
    this.source = source
    this.tokens = tokens
    this.i = 0
    this.skipNewlineStack = []
  }

  bool shouldSkipNewlines() {
    return this.skipNewlineStack and this.skipNewlineStack[-1]
  }

  SkippingNewlines skippingNewlines(bool skip) {
    return SkippingNewlines(this, skip)
  }

  ast.Token gettok() {
    final token = this.peek()
    this.i = this.i + 1
    return token
  }

  ast.Token peekAt(int offset) {
    if (this.shouldSkipNewlines()) {
      while (this.i < this.tokens.size and
             this.tokens[this.i].(ast.Token).type == '\n') {
        this.i = this.i + 1
      }
    }
    return this.tokens[this.i + offset].(ast.Token)
  }

  ast.Token peek() {
    return this.peekAt(0)
  }

  bool consume(String type) {
    if (this.at(type)) {
      this.gettok()
      return true
    }
    return false
  }

  bool consumeDelimiter() {
    if (this.atDelimiter()) {
      this.expectDelimiter()
      return true
    }
    return false
  }

  bool atDelimiterAt(int offset) {
    return this.typeAtIs(offset, ';') or this.typeAtIs(offset, '\n')
  }

  bool atDelimiter() {
    return this.atDelimiterAt(0)
  }

  ast.Token expectDelimiter() {
    final token = this.peek()
    if (!this.consume(';')) {
      this.expect('\n')
    }
    while (this.consume(';') or this.consume('\n')) {}
    return token
  }

  bool typeAtIs(int pos, String type) {
    return this.peekAt(pos).type == type
  }

  bool at(String type) {
    return this.typeAtIs(0, type)
  }

  ast.Token expect(String type) {
    if (!this.at(type)) {
      panic('Expected ' + type)
    }
    return this.gettok()
  }

  Map parseImport() {
    final token = this.peek()
    final parts = [this.expect('NAME').data]
    while (this.consume('.')) {
      parts.push(this.expect('NAME').data)
    }
    final alias = (
      this.consume('as') ?
        this.expect('NAME').data.(String) :
        parts[-1].(String)
    )
    this.expectDelimiter()
    return ast.Import(token, '.'.join(parts), alias)
  }

  Map parseTranslationUnit() {
    final token = this.peek()
    final imports = []
    this.consumeDelimiter()
    while (this.at('import')) {
      imports.push(this.parseImport())
    }
    final definitions = []
    return ast.TranslationUnit(token, this.source.name, imports, definitions)
  }
}

class SkippingNewlines {
  Parser parser
  bool skip

  new(Parser parser, bool skip) {
    this.parser = parser
    this.skip = skip
  }

  int Enter() {
    this.parser.skipNewlineStack.push(this.skip)
    return 0
  }

  void Exit() {
    this.parser.skipNewlineStack.pop()
  }
}
