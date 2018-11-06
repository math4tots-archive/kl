import typing


class Source(typing.NamedTuple):
    name: str
    path: typing.Optional[str]
    data: str


class Token(typing.NamedTuple):
    source: Source
    i: int
    type: str
    data: object

    def __repr__(self):
        return f'Token({repr(self.type)}, {repr(self.data)})'

    @property
    def lineno(self):
        return self.source.data.count('\n', self.i) + 1

    @property
    def colno(self):
        cn = 0
        i = self.i
        while i >= 0 and self.source.data[i] != '\n':
            i -= 1
            cn += 1
        return cn

    @property
    def line(self):
        s = self.source.data
        a = self.i
        while a > 0 and s[a - 1] != '\n':
            a -= 1
        b = self.i
        while b < len(s) and s[b] != '\n':
            b += 1
        return s[a:b]

    def format(self):
        return (
            f'  On line {self.lineno}\n'
            f'  {self.line}\n'
            f'  {" " * (self.colno - 1)}*\n'
        )


class Error(Exception):
    def __init__(self, tokens, message):
        super().__init__(f'{message}\n{"".join(t.format() for t in tokens)}')
        self.tokens = tokens
        self.message = message


KEYWORDS = {
  'is', 'not', 'null', 'true', 'false', 'new', 'and', 'or', 'in',
  'inline', 'extern', 'class', 'trait', 'final', 'def', 'auto',
  'for', 'if', 'else', 'while', 'break', 'continue', 'return',
  'with', 'from', 'import', 'as', 'try', 'catch', 'finally',
}

SYMBOLS = tuple(reversed(sorted([
  '\n',
  '||', '&&', '|', '&', '<<', '>>', '~',
  ';', '#', '?', ':', '!', '++', '--', '**',
  '.', ',', '!', '@', '^', '&', '+', '-', '/', '%', '*', '.', '=', '==', '<',
  '>', '<=', '>=', '!=', '(', ')', '{', '}', '[', ']',
])))

ESCAPE_MAP = {
  'n': '\n',
  't': '\t',
  'r': '\r',
  '\\': '\\',
  '"': '"',
  "'": "'",
}


def lex_string(data: str):
    return lex(Source('#', None, data))


def lex(source: Source):
    return list(lex_gen(source))


def lex_gen(source: Source):
    s = source.data
    i = 0

    def skip_spaces_and_comments():
        nonlocal i
        while s.startswith('//', i) or i < len(s) and s[i] in ' \t\r':
            if s.startswith('//', i):
                while i < len(s) and s[i] != '\n':
                    i += 1
            else:
                i += 1

    def mt(i, type_, data):
        # make token
        return Token(source, i, type_, data)

    def error(indices, message):
        tokens = [Token(source, i, 'ERR', None) for i in indices]
        return Error(tokens, message)

    while True:
        skip_spaces_and_comments()
        if i >= len(s):
            yield mt(i, 'EOF', 'EOF')
            break

        a = i

        # raw string literal
        if s.startswith(('r"', "r'"), i):
            i += 1  # 'r'
            q = s[i:i+3] if s.startswith(s[i] * 3, i) else s[i]
            i += len(q)
            while i < len(s) and not s.startswith(q, i):
                i += 1
            if i >= len(s):
                raise error([a], 'Unterminated raw string literal')
            i += len(q)
            yield mt(a, 'STRING', s[a+1+len(q):i-len(q)])
            continue

        # normal string literal
        if s.startswith(('"', "'"), i):
            q = s[i:i+3] if s.startswith(s[i] * 3, i) else s[i]
            i += len(q)
            sb = []
            while i < len(s) and not s.startswith(q, i):
                if s[i] == '\\':
                    i += 1
                    if i >= len(s):
                        raise error([a], 'Expected string escape')
                    if s[i] in ESCAPE_MAP:
                        sb.append(ESCAPE_MAP[s[i]])
                        i += 1
                    else:
                        raise error([i], 'Invalid string escape')
                else:
                    sb.append(s[i])
                    i += 1
            if i >= len(s):
                raise error([a], 'Unterminated string literal')
            i += len(q)
            yield mt(a, 'STRING', ''.join(sb))
            continue

        # numbers (int and float)
        if s[i].isdigit():
            while i < len(s) and s[i].isdigit():
                i += 1
            if i < len(s) and s[i] == '.':
                i += 1
                while i < len(s) and s[i].isdigit():
                    i += 1
                yield mt(a, 'FLOAT', float(s[a:i]))
            else:
                yield mt(a, 'INT', int(s[a:i]))
            continue

        # escaped names
        if s[i] == '`':
            i += 1
            while i < len(s) and s[i] != '`':
                s[i] += 1
            if i >= len(s):
                raise error([a], 'Unterminated escape identifier')
            i += 1
            yield mt(a, 'NAME', s[a+1:i-1])
            continue

        # names and keywords
        if s[i].isalnum() or s[i] == '_':
            while i < len(s) and s[i].isalnum() or s[i] == '_':
                i += 1
            name = s[a:i]
            type_ = name if name in KEYWORDS else 'NAME'
            yield mt(a, type_, name)
            continue

        # symbols
        for symbol in SYMBOLS:
            if s.startswith(symbol, i):
                i += len(symbol)
                yield mt(a, symbol, symbol)
                break
        else:
            # unrecognized token
            raise error([i], 'Unrecognized token')




print(lex_string('hello world!'))