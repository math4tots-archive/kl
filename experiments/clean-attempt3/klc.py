import contextlib
import os
import typing


_scriptdir = os.path.dirname(os.path.realpath(__file__))


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

    @classmethod
    def load(cls, name, tokens=None):
        """Load KL source with given name
        """
        relpath = cls._name_to_relative_path(name)
        for root in cls.get_source_roots():
            path = os.path.join(root, relpath)
            if os.path.isfile(path):
                return cls.load_from_path(path, name=name, tokens=tokens)
        raise Error(tokens or [], f'Could not find module {name} ({relpath})')

    @classmethod
    def load_from_path(cls, path, name=None, tokens=None):
        with open(path) as f:
            data = f.read()
        return Source(name, path, data)

    @classmethod
    def _name_to_relative_path(cls, name):
        return os.path.join(*name.split('.')) + '.k'

    @classmethod
    def get_source_roots(cls):
        yield os.path.join(_scriptdir, 'lib')


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
                while i < len(s) and (s[i].isalnum() or s[i] == '_'):
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
    def load_from_path(path, name=None, tokens=None):
        return parser.parse(Source.load_from_path(
            path, name=name, tokens=tokens))

    module_cache = dict()

    @ns
    def load(name, tokens=None):
        if name not in module_cache:
            module_cache[name] = parser.parse(Source.load(name, tokens=tokens))
        return module_cache[name]

    NT = typing.NamedTuple

    @ns
    class TranslationUnit(NT):
        metadata: dict
        name: str
        imports: list
        definitions: list

    @ns
    class Import(NT):
        name: str
        alias: str

    @ns
    class FunctionDefinition(NT):
        return_type: str
        name: str
        parameters: list
        body: typing.Optional[object]

    @ns
    class Parameter(NT):
        type: str
        name: str

    @ns
    class Block(NT):
        statements: list

    @ns
    class PrimitiveTypeDefinition(NT):
        name: str
        cname: str
        methods: list

    @ns
    class TraitDefinition(NT):
        name: str
        traits: list
        methods: list

    @ns
    class ClassDefinition(NT):
        extern: bool
        name: str
        traits: list
        fields: typing.Optional[list]
        methods: list

    @ns
    class GlobalVariableDefinition(NT):
        type: str
        name: str
        expression: typing.Optional[object]

    type_definition_types = (
        PrimitiveTypeDefinition,
    )

    ns(type_definition_types, name='type_definition_types')


@Namespace
def parser(ns):
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
                i += 2
            elif at('class', '*', 'NAME'):
                names.add(tokens[i + 2].value)
                i += 3
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
                raise error(
                    [i],
                    f'Module alias and exported names are not allowed here')
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
                pname = expect_clean_name()
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

        def parse_type_body(*, allow_fields):
            expect('{')
            consume('DELIM')
            methods = dict()
            fields = [] if allow_fields else None
            while not consume('}'):
                if at_function_definition():
                    methods.append(parse_function_definition())
                elif allow_fields:
                    field_token = peek()
                    field_type = parse_type()
                    field_name = expect_clean_name()
                    expect('DELIM')
                    fields.append(
                        ast.Field(field_token, field_type, field_name))
                else:
                    raise error([i], 'Expected method definition')
            expect('DELIM')
            return methods, fields

        def parse_class():
            token = peek()
            extern = consume('extern')

            trait = not extern and consume('trait')
            if not trait:
                expect('class')

            primitive = not extern and not trait and consume('*')

            name = expect('NAME').value

            if primitive:
                cname = expect('STRING').value
            else:
                traits = []
                if consume('('):
                    with nls(True):
                        while not consume(')'):
                            traits.append(parse_type())
                            if not consume(','):
                                expect(')')
                                break

            methods, fields = parse_type_body(
                allow_fields=not extern and not trait and not primitive)

            if primitive:
                cls = ast.PrimitiveTypeDefinition(name, cname, methods)
            elif trait:
                cls = ast.TraitDefinition(name, traits, methods)
            else:
                cls = ast.ClassDefinition(
                    extern, name, traits, fields, methods)

            return wtok(token, cls)


        def parse_global_definition():
            if at_function_definition():
                return parse_function_definition()
            elif at('class') or seq(['extern', 'class']) or at('trait'):
                return parse_class()
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
            name = '.'.join(parts)
            for part in parts:
                for c in part:
                    if not (ord('a') <= ord(c) <= ord('z')):
                        raise Error(
                            [token],
                            'Module names must only contain lowercase letters')
            imports.append(wtok(token, ast.Import(name, alias)))
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
        return parse(Source(None, None, s))


@Namespace
def annotator(ns):

    class Meta(typing.NamedTuple):
        data: dict

        def get(self, node):
            return self.data[id(node)]

        def gettok(self, node):
            return self.get(node)['@token']

    headers_cache = dict()
    loaded_module_names = set()
    annotation_cache = dict()

    def load_headers(name):
        if name not in loaded_module_names:
            loaded_module_names.add(name)
            load_headers_for_translation_unit(ast.load(name))

    def load_headers_for_translation_unit(tu: ast.TranslationUnit):
        for imp in tu.imports:
            load_headers(imp.name)

        for defn in tu.definitions:
            if defn.name in headers_cache:
                raise Error(
                    [defn.token, headers_cache[defn.name].token],
                    f'Duplicate definition {defn.name}')
            headers_cache[defn.name] = defn

    def load_headers_from_path(path):
        load_headers_for_translation_unit(ast.load_from_path(path))

    def get_defn(name, tokens=None):
        if name not in headers_cache:
            raise Error(tokens or [], f'"{name}" is not defined')
        return headers_cache[name]

    def assert_any_type(name, tokens=None):
        tokens = tokens or []
        defn = get_defn(name)
        if type(defn) not in ast.type_definition_types:
            raise Error(tokens + [defn.token], f'"{name}" is not a type')

    def assert_var_type(name, tokens=None):
        if name == 'void':
            raise Error(tokens or [], f'void type not allowed here')
        assert_any_type(name, tokens)

    @ns
    def load(name):
        if name not in annotation_cache:
            tu = ast.load(name)
            annotate(tu)
            annotation_cache[name] = tu
        return annotation_cache[name]

    @ns
    def load_from_string(s):
        tu = parser.parse_string(s)
        annotate(tu)
        return tu

    @ns
    def annotate(tu: ast.TranslationUnit):
        load_headers_for_translation_unit(tu)
        meta = Meta(tu.metadata)

        for defn in tu.definitions:
            tp = type(defn)
            if tp == ast.FunctionDefinition:
                annotate_function_definition(defn, meta)
            else:
                raise Error(
                    [defn.token],
                    f'Unrecognized global definition type {tp}')

    @ns
    def annotate_function_definition(fd: ast.FunctionDefinition, meta):
        assert_any_type(fd.return_type)
        for param in fd.parameters:
            assert_var_type(param.type, tokens=[meta.gettok(param)])

    # We initialize the annotator with prelude.k
    load_headers_from_path(os.path.join(_scriptdir, 'prelude.k'))


@Namespace
def translator(ns):

    @ns
    def translate(tu: ast.TranslationUnit, basename=None):
        if basename is None:
            basename = tu.name.replace('.', os.sep)
        dirpath = os.path.join(_scriptdir, 'generated')
        header_path = os.path.join(dirpath, f'klcn{basename}.h')
        source_path = os.path.join(dirpath, f'klcn{basename}.c')
        os.makedirs(dirpath, exist_ok=True)
        hdr = translate_header(tu)
        src = translate_source(tu)
        with open(header_path, 'w') as f:
            f.write(hdr)
        with open(source_path, 'w') as f:
            f.write(src)

    def translate_header(tu):
        return ''

    def translate_source(tu):
        return ''


tu = annotator.load_from_string("""
import os

int foo(int x) {
}

void main() {
}
""")

translator.translate(tu, basename='main')

print(ast)
print(ast.TranslationUnit)
