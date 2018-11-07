import contextlib
import typing


class Namespace:
    def __init__(self, f):
        object.__setattr__(self, 'attrs', dict())
        object.__setattr__(self, 'name', f.__name__)
        f(self)

    def __getattribute__(self, key):
        attrs = object.__getattribute__(self, 'attrs')
        if key not in attrs:
            raise AttributeError(
                f'No attr {key} for {object.__getattribute__(self, "name")}')
        return attrs[key]

    def __setattr__(self, key, value):
        object.__getattribute__(self, 'attrs')[key] = value

    def __call__(self, x, name=None):
        if name is None:
            name = x.__name__
        setattr(self, name, x)
        return x

    def __repr__(self):
        return f'<namespace {object.__getattribute__(self, "name")}>'


class Source(typing.NamedTuple):
    name: str
    path: typing.Optional[str]
    data: str


class Token(typing.NamedTuple):
    source: Source
    i: int
    type: str
    value: object

    def __repr__(self):
        return f'Token({repr(self.type)}, {repr(self.value)})'

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


@Namespace
def lexer(ns):
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

    @ns
    def lex_string(data: str):
        return lex(Source('#', None, data))

    @ns
    def lex(source: Source):
        return list(lex_gen(source))

    def lex_gen(source: Source):
        s = source.data
        i = 0
        is_delim = False

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

            a = i
            last_was_delim = is_delim
            is_delim = False

            if i >= len(s):
                if not last_was_delim:
                    yield mt(i, 'DELIM', 'EOF')
                yield mt(i, 'EOF', 'EOF')
                break

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

            # delimiters (';' and '\n')
            if s[i] == ';' or s[i] == '\n':
                i += 1
                is_delim = True
                if not last_was_delim:
                    yield mt(a, 'DELIM', s[i - 1])
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


@Namespace
def ast(ns):

    @ns
    class TranslationUnit(typing.NamedTuple):
        metadata: dict
        name: str
        imports: list
        definitions: list

    @ns
    class Import(typing.NamedTuple):
        name: str
        alias: str

    @ns
    class FunctionDefinition(typing.NamedTuple):
        return_type: 'type'
        name: 'name'
        parameters: list
        body: typing.Optional[object]

    @ns
    class Parameter(typing.NamedTuple):
        type: 'type'
        name: 'name'

    @ns
    class Block(typing.NamedTuple):
        statements: list


@Namespace
def parser(ns):
    def resolve_names(tu: ast.TranslationUnit):
        """
        Resolves:
            * import alias qualified names to fully qualified names, and
            * qualifies exported names based on this module's name
        """
        metadata = dict()
        import_map = {i.alias: i.name for i in tu.imports}
        local_map = {
            df.name: f'{tu.name}.{df.name}' for df in tu.definitions
        } if tu.name else dict()
        definitions = []

        def wm(old_node, new_node):
            # wm = with metadata (from old_node)
            m[id(new_node)] = tu.metadata[id(old_node)]
            return new_node

        def error(nodes, message):
            tokens = [tu.metadata[id(n)]['@token'] for n in nodes]
            return Error(tokens, message)

        def resolve_name(name):
            if '.' in name and name.split('.')[0] in import_map:
                module_alias, short_name = name.split('.')
                return import_map[module_alias] + '.' + short_name
            elif name in local_map:
                return local_map[name]
            else:
                return name

        def resolve(node):
            if type(node) is list:
                return [resolve(x) for x in node]
            elif isinstance(node, tuple) and type(node) != tuple:
                fields = []
                infos = type(node).__annotations__.items()
                for value, (_, type_) in zip(node, infos):
                    if type_ in ('type', 'name'):
                        fields.append(resolve_name(value))
                    else:
                        fields.append(resolve(value))
                return wm(node, type(node)(*fields))
            elif isinstance(node, (int, float, str, type(None), bool)):
                return node
            else:
                raise error([], f'Unrecognized node type: {node}')

        for df in tu.definitions:
            definitions.append(resolve(df))

        return wm(tu, ast.TranslationUnit(m, tu.name, tu.imports, definitions))

    def peek_exported_names(tokens: typing.List[Token], i):
        """
        Try to figure out what names will be declared in this file
        before actually parsing.
        This is used to handle namespacing -- so that the given names
        when referenced will be properly qualified when used.
        """
        types = [token.type for token in tokens]
        vals = [token.value for token in tokens]
        names = set()

        def at(*seq):
            return tuple(types[i:i+len(seq)]) == seq

        def skip_grouping(open, close):
            nonlocal i
            assert types[i] == open
            assert open != close
            depth = 1
            start = i
            i += 1
            while depth and i < len(tokens):
                if types[i] == open:
                    depth += 1
                elif types[i] == close:
                    depth -= 1
                i += 1
            if depth:
                raise Error([tokens[start]], f'Mismatched grouping token')

        while i < len(tokens):
            if at('NAME' ,'NAME') or at('class', 'NAME') or at('trait', 'NAME'):
                # function or variable or class or trait definition
                names.add(tokens[i + 1].value)
                i += 1
            elif types[i] == '(':
                skip_grouping('(', ')')
            elif types[i] == '[':
                skip_grouping('[', ']')
            elif types[i] == '{':
                skip_grouping('{', '}')
            else:
                i += 1

        return names

    @ns
    def parse(source: Source):
        m = dict()
        i = 0
        ts = lexer.lex(source)
        nlstack = []
        imports = []
        definitions = []
        module_map = dict()
        scope_stack = [dict()]

        def scope_put(key, type_):
            scope_stack[-1]

        def error(indices, message):
            tokens = [ts[i] for i in indices]
            return Error(tokens, message)

        def should_skip_newlines():
            return nlstack and nlstack[-1]

        @contextlib.contextmanager
        def nls(skip):
            # nls = new line skip
            nlstack.append(skip)
            yield
            nlstack.pop()

        def peek(offset=0):
            nonlocal i
            if should_skip_newlines():
                while i < len(ts) and ts[i].type == '\n':
                    i += 1
            return ts[i + offset]

        def gettok():
            nonlocal i
            token = peek()
            i += 1
            return token

        def at(tp, offset=0):
            return (
                peek(offset).type in tp if isinstance(tp, tuple) else
                peek(offset).type == tp
            )

        def seq(tps, offset=0):
            return all(at(tp, offset + i) for i, tp in enumerate(tps))

        def consume(tp):
            if at(tp):
                return gettok()

        def expect(tp):
            if not at(tp):
                raise error([i], f'Expected {tp} but got {peek()}')
            return gettok()

        def mark_identifiers(new_type, name_set):
            """Changes the token type of any NAME whose value is in name_set
            to new_type
            """
            for j in range(i, len(ts)):
                if ts[j].type == 'NAME' and ts[j].value in name_set:
                    t = ts[j]
                    ts[j] = Token(t.source, t.i, new_type, t.value)

        def mark_module_identifiers():
            mark_identifiers(new_type='MODULE', name_set=module_map)

        def mark_exported_identifiers():
            mark_identifiers(new_type='EXPORT', name_set=exported_names)

        def qualify_exported_name(name):
            if source.name:
                return f'{source.name}.{name}'
            else:
                return name

        def rename_identifiers():
            """Convert all MODULE and EXPORT tokens to their corresponding
            NAME tokens. This way, when doing the parse, we can treat
            them like normal NAME tokens.
            """
            nonlocal ts
            new_tokens = ts[:i]
            j = i
            while j < len(ts):
                if ts[j].type == 'MODULE':
                    if (j + 2 >= len(ts) or
                            ts[j + 1].type != '.' or
                            ts[j + 2].type != 'NAME'):
                        raise Error(
                            [ts[j]],
                            'Module aliases must be used only to qualify '
                            'members of the module and cannot be used '
                            'as normal identifiers')
                    new_tokens.append(Token(
                        ts[j].source,
                        ts[j].i,
                        'NAME',
                        f'{module_map[ts[j].value]}.{ts[j + 2].value}'
                    ))
                    j += 3
                elif ts[j].type == 'EXPORT':
                    new_tokens.append(Token(
                        ts[j].source,
                        ts[j].i,
                        'NAME',
                        qualify_exported_name(ts[j].value),
                    ))
                    j += 1
                else:
                    new_tokens.append(ts[j])
                    j += 1
            ts = new_tokens

        def md(node):
            # metadeta dictionary for node
            key = id(node)
            if key not in m:
                m[key] = dict()
            return m[key]

        def wtok(tok, node):
            # with token
            md(node)['@token'] = tok
            return node

        def expect_clean_name():
            """A clean name is a NAME that does not have a '.' in the name
            This is to distinguish between global qualified names
            and normal names.
            """
            if not at('NAME'):
                expect('NAME')
            if '.' in peek().value:
                raise error([i], f'Expected clean name but got {peek()}')
            return expect('NAME').value

        def parse_type():
            return parse_identifier()

        def parse_identifier():
            return expect('NAME').value

        def at_variable_definition():
            return (
                seq(['NAME', 'NAME', 'DELIM']) or
                seq(['NAME', 'NAME', '='])
            )

        def at_function_definition():
            return (
                seq(['NAME', 'NAME', '('])
            )

        def parse_function_definition():
            token = peek()
            return_type = parse_type()
            name = expect('NAME').value
            params = []
            expect('(')
            while not consume(')'):
                ptoken = peek()
                ptype = parse_type()
                pname = expect('NAME').value
                params.append(wtok(ptoken, ast.Parameter(ptype, pname)))
                if not consume(','):
                    expect(')')
                    break
            if consume('DELIM'):
                body = None
            else:
                body = parse_block()
            return wtok(token, ast.FunctionDefinition(
                return_type, name, params, body))

        def parse_global_definition():
            if at_function_definition():
                return parse_function_definition()
            raise error([i], 'Expected global definition')

        def parse_block():
            token = expect('{')
            statements = []
            consume('DELIM')
            while not consume('}'):
                statements.append(parse_statement())
            expect('DELIM')
            return wtok(token, ast.Block(statements))

        def verify_peek_exported_names():
            # After the real parse as finished,
            # we should compare the result peeked exported names
            # from peek_exported_names with the real parsed
            # results in 'definitions'
            qualified_names = {df.name for df in definitions}
            qualified_names0 = {
                qualify_exported_name(n) for n in exported_names
            }
            # If this assertion fails, it means there's an issue with
            # the approximation logic in peek_exported_names(..)
            assert qualified_names0 == qualified_names, [
                qualified_names0 - qualified_names,
                qualified_names - qualified_names0,
            ]

        consume('DELIM')
        while at('import'):
            token = expect('import')
            parts = [expect('NAME').value]
            while consume('.'):
                parts.append(expect('NAME').value)
            alias = expect('NAME').value if consume('as') else parts[-1]
            imports.append(wtok(token, ast.Import('.'.join(parts), alias)))
            expect('DELIM')

        for imp in imports:
            module_map[imp.alias] = imp.name

        # The following 4 lines are a hack to make namespaces work
        mark_module_identifiers()
        exported_names = peek_exported_names(ts, i)
        mark_exported_identifiers()
        rename_identifiers()

        while not at('EOF'):
            definitions.append(parse_global_definition())

        verify_peek_exported_names()

        return ast.TranslationUnit(m, source.name, imports, definitions)

    @ns
    def parse_string(s: str):
        return parse(Source('#', None, s))


print(parser.parse_string("""
import os
import klc.lexer as lx
import klc.parser

os.Path foo(int x, os.Path p) {
}

void main() {
}
"""))
print(ast)
print(ast.TranslationUnit)
