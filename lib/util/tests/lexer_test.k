import util.lexer


void main() {
  final source = lexer.Source('source name', 'contents')
  assert(str(source) == '<Source "source name">')
}
