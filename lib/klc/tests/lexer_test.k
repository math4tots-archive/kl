import klc.lexer
import klc.ast

List lex(String data) {
  return lexer.lex(ast.Source('<test>', data))
}

List lex2(String data) {
  return lex(data).map(def List[](ast.Token token) {
    return [token.type, token.data]
  })
}

void main() {
  assert(lexer.isdigit('0'))
  assert(lexer.isdigit('1'))
  assert(lexer.isdigit('2'))

  assert(lex2('10 20 30.3') == [
    ["INT", "10"], ["INT", "20"], ["FLOAT", "30.3"], ["EOF", null]
  ])
  assert(lex2("""
  for x
  """) == [
    ["\n", "\n"],
    ["for", "for"],
    ["NAME", "x"],
    ["\n", "\n"],
    ["EOF", null]]
  )

  assert(lex2("`name`") ==  [["NAME", "name"], ["EOF", null]])

  realLifeExample()
}

void realLifeExample() {
  assert(lex2(r"""
    if (!this.more()) {
      return Try(false, ast.Error(
        [this.mktok(this.i, 'ERR', null)],
        'Expected string literal escape'
      ))
    }
  """) == [
    ["\n", "\n"],
    ["if", "if"],
    ["(", "("],
    ["!", "!"],
    ["NAME", "this"],
    [".", "."],
    ["NAME", "more"],
    ["(", "("],
    [")", ")"],
    [")", ")"],
    ["{", "{"],
    ["\n", "\n"],
    ["return", "return"],
    ["NAME", "Try"],
    ["(", "("],
    ["false", "false"],
    [",", ","],
    ["NAME", "ast"],
    [".", "."],
    ["NAME", "Error"],
    ["(", "("],
    ["\n", "\n"],
    ["[", "["],
    ["NAME", "this"],
    [".", "."],
    ["NAME", "mktok"],
    ["(", "("],
    ["NAME", "this"],
    [".", "."],
    ["NAME", "i"],
    [",", ","],
    ["STRING", "ERR"],
    [",", ","],
    ["null", "null"],
    [")", ")"],
    ["]", "]"],
    [",", ","],
    ["\n", "\n"],
    ["STRING", "Expected string literal escape"],
    ["\n", "\n"],
    [")", ")"],
    [")", ")"],
    ["\n", "\n"],
    ["}", "}"],
    ["\n", "\n"],
    ["EOF", null]])
}
