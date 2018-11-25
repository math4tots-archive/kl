import argparse
import contextlib
import itertools
import os
import shutil
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


class Multimethod:
    # TODO: support inheritance
    def __init__(self, name, n=1):
        # n = number of arguments whose types
        # we take into account for dispatch
        self.name = name
        self.n = n
        self.table = dict()

    def __repr__(self):
        return f'Multimethod({self.name}, {self.n})'

    def on(self, *types):
        if len(types) != self.n:
            raise TypeError(f'self.n = {self.n}, types = {types}')
        def wrapper(f):
            self.table[types] = f
            return self
        return wrapper

    def find(self, types):
        "Find and return the implementation for the given arg types"
        if types not in self.table:
            mrolist = [t.__mro__ for t in types]
            for basetypes in itertools.product(*mrolist):
                if basetypes in self.table:
                    self.table[types] = self.table[basetypes]
                    break
            else:
                raise KeyError(
                    f'{repr(self.name)} is not defined for types {types}')
        return self.table[types]

    def __call__(self, *args):
        types = tuple(type(arg) for arg in args[:self.n])
        f = self.find(types)
        return f(*args)


class Source(typing.NamedTuple):
    filename: str
    data: str

    @classmethod
    def from_path(cls, path):
        with open(path) as f:
            data = f.read()
        return cls(path, data)


class Token(typing.NamedTuple):
    source: Source
    i: int
    type: str
    value: object

    def __repr__(self):
        return f'Token({repr(self.type)}, {repr(self.value)})'

    @property
    def lineno(self):
        return self.source.data.count('\n', 0, self.i) + 1

    @property
    def colno(self):
        s = self.source.data
        a = self.i
        while a > 0 and s[a - 1] != '\n':
            a -= 1
        return self.i - a + 1

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
            f'  In {self.source.filename} on line {self.lineno}\n'
            f'  {self.line}\n'
            f'  {" " * (self.colno - 1)}*\n'
        )


class BaseError(Exception):
    def __init__(self, tokens, message):
        super().__init__(f'{message}\n{"".join(t.format() for t in tokens)}')
        self.tokens = tuple(tokens)
        self.message = message


class Error(BaseError):
    pass


class Fubar(BaseError):
    pass


class Promise:
    def __init__(self, callback):
        self._callback = callback
        self._resolved = False
        self._result = None

    def resolve(self):
        if not self._resolved:
            self._result = self._callback()
        return self._result

    def __repr__(self):
        return f'Promise({repr(self.resolve())})'


class Node(object):
    def __init__(self, token, *args):
        self.token = token
        for (fname, ftype), arg in zip(type(self).fields, args):
            if not isinstance(arg, ftype):
                raise TypeError('Expected type of %r to be %r, but got %r' % (
                    fname, ftype, arg))
            setattr(self, fname, arg)
        if len(type(self).fields) != len(args):
            raise TypeError('%s expects %s arguments, but got %s' % (
                type(self).__name__, len(type(self).fields), len(args)))

    def __repr__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join(repr(getattr(self, n)) for n, _ in type(self).fields),
        )

    @classmethod
    def map(cls, f, node):
        nt = type(node)
        return nt(node.token, *[
            cls._map_helper(getattr(node, fieldname))
            for fieldname, _ in nt.fields
        ])

    @classmethod
    def _map_helper(cls, f, value):
        if isinstance(value, (type(None), int, float, bool, str, IR.CType)):
            return value
        if isinstance(value, list):
            return [cls._map_helper(x) for x in value]
        if isinstance(value, tuple):
            return tuple(cls._map_helper(x) for x in value)
        if isinstance(value, set):
            return {cls._map_helper(x) for x in value}
        if isinstance(value, dict):
            return {
                cls._map_helper(k): cls._map_helper(v)
                for k, v in value.items()
            }
        if isinstance(value, Node):
            return f(cls.map(f, value))
        raise TypeError(
            f'Unsupported Node.map element type '
            f'{type(value)} ({repr(value)})')


@Namespace
def typeutil(ns):
    class FakeType(object):
        @property
        def __name__(self):
            return repr(self)

    class ListType(FakeType):
        def __init__(self, subtype):
            self.subtype = subtype

        def __getitem__(self, subtype):
            return ListType(subtype)

        def __instancecheck__(self, obj):
            return (
                isinstance(obj, list) and
                all(isinstance(x, self.subtype) for x in obj))

        def __repr__(self):
            return 'List[%s]' % (self.subtype.__name__, )

    List = ListType(object)
    ns(List, 'List')

    class OptionalType(FakeType):
        def __init__(self, subtype):
            self.subtype = subtype

        def __getitem__(self, subtype):
            return OptionalType(subtype)

        def __instancecheck__(self, obj):
            return obj is None or isinstance(obj, self.subtype)

        def __repr__(self):
            return 'Optional[%s]' % (self.subtype.__name__, )

    Optional = OptionalType(object)
    ns(Optional, 'Optional')


@Namespace
def lexer(ns):
    KEYWORDS = {
      'is', 'not', 'null', 'true', 'false', 'new', 'and', 'or', 'in',
      'inline', 'extern', 'class', 'trait', 'final', 'def', 'auto',
      'struct',
      'for', 'if', 'else', 'while', 'break', 'continue', 'return',
      'with', 'from', 'import', 'as', 'try', 'catch', 'finally', 'raise',
      'except', 'case','switch', 'var',
    }
    ns(KEYWORDS, 'KEYWORDS')

    SYMBOLS = tuple(reversed(sorted([
      '\n',
      '||', '&&', '|', '&', '<<', '>>', '~', '...',
      ';', '#', '?', ':', '!', '++', '--',  # '**',
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
        return lex(Source(None, data))

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
                    yield mt(i, '\n', 'EOF')
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
                    i += 1
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
def IR(ns):
    "Intermediate Representation"

    BUILTIN = lexer.lex(Source('builtin', 'builtin'))[0]

    @ns
    class Type:
        pass

    @ns
    class Stub(Node):
        """Stubs stand in as declarations for names.
        They are the things you get when you lookup a scope.
        We use stubs instead of full definitions because
        we need something like this in order to parse the full
        definition in the first place.

        abstract fields/methods:

            token: Token
        """

    class PrimitiveType(Type, Stub):
        token = BUILTIN

        def __init__(self, name):
            self.name = name

        @property
        def key(self):
            return (type(self), self.name)

        def __hash__(self):
            return hash(self.key)

        def __eq__(self, other):
            return type(self) is type(other) and self.key == other.key

        def __repr__(self):
            return f'PrimitiveType({self.name})'

    PRIMITIVE_TYPE_NAMES = (
        'void',
        'int',
        'double',
    )

    PRIMITIVE_TYPE_MAP = {
        name: PrimitiveType(name) for name in PRIMITIVE_TYPE_NAMES
    }
    ns(PRIMITIVE_TYPE_MAP, 'PRIMITIVE_TYPE_MAP')

    VOID = PRIMITIVE_TYPE_MAP['void']
    ns(VOID, 'VOID')

    class N(Node):
        pass

    @ns
    class Statement(N):
        pass

    @ns
    class Expression(N):
        pass

    @ns
    class GlobalDefinition(N):
        pass

    @ns
    class Import(N):
        fields = (
            ('module_name', str),
        )

    @ns
    class TranslationUnit(N):
        fields = (
            ('imports', typeutil.List[Import]),
            ('definitions', typeutil.List[GlobalDefinition]),
        )

    @ns
    class StructField(N):
        fields = (
            ('extern', bool),
            ('type_promise', Promise),
            ('name', str),
        )

        @property
        def type(self):
            return self.type_promise.resolve()

    @ns
    class StructDefinition(GlobalDefinition, Type, Stub):
        fields = (
            ('extern', bool),
            ('name', str),
            ('fields', typeutil.List[StructField]),
        )

    @ns
    class Parameter(N, Stub):
        fields = (
            ('type_promise', Promise),
            ('name', str),
        )

        @property
        def type(self):
            return self.type_promise.resolve()

    @ns
    class FunctionDefinition(GlobalDefinition, Stub):
        fields = (
            ('extern', bool),
            ('return_type_promise', Promise),
            ('name', str),
            ('params', typeutil.List[Parameter]),
            ('vararg', bool),
            ('body_promise', Promise),
        )

        @property
        def return_type(self):
            return self.return_type_promise.resolve()

        @property
        def body(self):
            self.body_promise.resolve()

    @ns
    class Block(Statement):
        fields = (
            ('stmts', typeutil.List[Statement]),
        )


@Namespace
def parser(ns):
    class Scope:
        "Used for parsing, but 'private' to parser"
        def __init__(self, parent):
            self.parent = parent
            self.table = dict()
            self.root = self if parent is None else parent.root

        def __getitem__(self, key: str) -> IR.Stub:
            if key in self.table:
                return self.table[key]
            elif self.parent is not None:
                return self.parent[key]
            else:
                raise self.error(f'{repr(key)} not defined')

        def __setitem__(self, key: str, stub: IR.Stub):
            assert isinstance(stub, IR.Stub), stub
            if key in self.table:
                oldtoken = self.table[key].token
                with scope.push(oldtoken), scope.push(stub.token):
                    raise error(f'Duplicate definition of {repr(key)}')
            self.table[key] = stub

        def __contains__(self, key: str):
            return key in self.table or self.parent and key in self.parent

        def get_type_with_name(self, key: str):
            stub = self[key]
            if not isinstance(stub, IR.Type):
                raise error(f'{repr(key)} is not a type')
            return stub

        def load_source(self, module_name) -> Source:
            if module_name not in self.root.sources_table:
                relpath = module_name.replace('.', os.path.sep) + '.k'
                path = os.path.join(self.root.search_dir, relpath)
                with open(path) as f:
                    data = f.read()
                self.root.sources_table[path] = (
                    Source(module_name, path, data)
                )
            return self.root.sources_table[module_name]

        def load_module_scope(self, module_name) -> 'Scope':
            self.load(module_name)
            return self.root.module_scope_table[module_name]

        def load(self, module_name) -> 'Promise':
            if module_name not in self.root.promise_table:
                module_scope = Scope(self.root)
                if module_name in self.root.module_stack:
                    raise self.error(f'Circular import with {module_name}')
                self.root.module_scope_table[module_name] = module_scope
                try:
                    self.root.module_stack.append(module_name)
                    self.root.promise_table[module_name] = (
                        parse_module(module_scope, module_name)
                    )
                finally:
                    self.root.module_stack.pop()
            return self.root.promise_table[module_name]

        @property
        def is_root(self):
            return self.root is self

        @contextlib.contextmanager
        def push(self, token):
            self.root.stack.append(token)
            try:
                yield
            finally:
                self.root.stack.pop()

        def error(self, message):
            return Error(self.root.stack, message)

        def fubar(self, message):
            return Fubar(self.stack, message)

    @ns
    def parse(
            sources_table: typing.Dict[str, Source],
            search_dir: str):
        # This should be the only place where the 'root' scope is
        # created.
        scope = Scope(None)
        scope.sources_table = sources_table
        scope.search_dir = search_dir
        scope.promise_table = dict()
        scope.module_scope_table = dict()
        scope.stack: typing.List[Token] = []
        scope.module_stack: typing.List[str] = []
        for name, primitive_type in IR.PRIMITIVE_TYPE_MAP.items():
            scope[name] = primitive_type

        for module_name in sources_table.keys():
            scope.load(module_name).resolve()

        return {
            module_name: scope.load(module_name).resolve()
            for module_name in tuple(scope.sources_table.keys())
        }

    def parse_module(scope: Scope, module_name: str) -> Promise:
        # Check that this is a module level scope
        assert scope.parent.is_root

        source = scope.load_source(module_name)
        tokens = lexer.lex(source)
        i = 0
        indent_stack = []

        def qualify(name):
            return f'{module_name}.{name}'

        def should_skip_newlines():
            return indent_stack and indent_stack[-1]

        @contextlib.contextmanager
        def skipping_newlines(skipping):
            indent_stack.append(skipping)
            yield
            indent_stack.pop()

        def peek():
            nonlocal i
            if should_skip_newlines():
                while i < len(tokens) and tokens[i].type == '\n':
                    i += 1
            return tokens[i]

        def gettok():
            nonlocal i
            token = peek()
            i += 1
            return token

        def at(t: str):
            return peek().type == t

        def consume(t: str):
            if at(t):
                return expect(t)

        def consume_all(t: str):
            while consume(t):
                pass

        def expect(t: str):
            if not at(t):
                with push(peek()):
                    raise error(f'Expected {t} but got {peek()}')
            return gettok()

        def expect_id():
            return expect('NAME').value

        def parse_import():
            token = peek()
            using_from = consume('from')
            if not using_from:
                expect('import')

            parts = [expect_id()]
            while consume('.'):
                parts.append(expect_id())

            module_name = '.'.join(parts)

            if using_from:
                expect('import')
                exported_name = expect_id()

            alias = (
                expect_id() if consume('as') else
                exported_name if using_from else
                parts[-1]
            )

            with scope.push(token):
                module_scope = scope.load_module_scope(module_name)

                if using_from:
                    exported_def = module_scope[exported_name]
                else:
                    exported_def = (
                        IR.ImportStub(token, dict(module_scope.table))
                    )

                scope[alias] = exported_def

            @Promise
            def promise():
                with scope.push(token):
                    scope.load(module_name).resolve()
                return IR.Import(token, module_name)

            return promise

        def make_member_type_promise(base_type_promise, member_name):
            @Promise
            def promise():
                return (
                    base_type_promise
                        .resolve()
                        .get_member_type_with_name(member_name)
                )
            return promise

        def make_pointer_type_promise(base_type_promise):
            @Promise
            def promise():
                return IR.PointerType(base_type_promise.resolve())
            return promise

        def make_const_type_promise(base_type_promise):
            @Promise
            def promise():
                return IR.ConstType(base_type_promise.resolve())
            return promise

        def parse_type():
            token = peek()
            name = expect_id()

            @Promise
            def promise():
                with scope.push(token):
                    return scope.get_type_with_name(name)

            while True:
                if consume('.'):
                    member_name = expect_id()
                    promise = make_member_type_promise(promise, member_name)
                elif consume('*'):
                    promise = make_pointer_type_promise(promise)
                elif consume('const'):
                    promise = make_const_type_promise(promise)
                else:
                    break

            return promise

        def parse_struct(token, extern):
            name = expect_id()
            qualified_name = name if extern else qualify(name)
            fields = []
            expect('{')
            consume_all('\n')
            while not consume('}'):
                field_token = peek()
                field_extern = bool(consume('extern'))
                field_type_promise = parse_type()
                field_name = expect_id()
                fields.append(IR.StructField(
                    field_token,
                    field_extern,
                    field_type_promise,
                    field_name,
                ))
                consume_all('\n')
            defn = IR.StructDefinition(
                token,
                extern,
                qualified_name,
                fields,
            )
            with scope.push(token):
                scope[name] = defn
            @Promise
            def promise():
                return defn
            return promise

        def expect_params():
            expect('(')
            params = []
            vararg = False
            with skipping_newlines(True):
                while not consume(')'):
                    if consume('...'):
                        vararg = True
                        expect(')')
                        break
                    param_token = peek()
                    param_type_promise = parse_type()
                    param_name = expect_id()
                    params.append(IR.Parameter(
                        param_token,
                        param_type_promise,
                        param_name,
                    ))
                    if not consume(','):
                        expect(')')
                        break
            return params, vararg

        def parse_function(token, extern, type_promise, name):
            qualified_name = name if extern else qualify(name)
            params, vararg = expect_params()
            fscope = Scope(scope)
            for param in params:
                fscope[param.name] = param
            body_promise = None if consume('\n') else parse_block(fscope)
            defn = IR.FunctionDefinition(
                token,
                extern,
                type_promise,
                qualified_name,
                params,
                vararg,
                body_promise,
            )
            scope[name] = defn
            @Promise
            def promise():
                with scope.push(token):
                    for param in defn.params:
                        param.type
                    defn.return_type
                    defn.body
                return defn
            return promise

        def parse_block(scope):
            token = expect('{')
            stmt_promises = []
            with skipping_newlines(False):
                consume_all('\n')
                while not consume('}'):
                    stmt_promises.append(parse_statement(scope))
                    consume_all('\n')
            @Promise
            def promise():
                return IR.Block(
                    token,
                    [p.resolve() for p in stmt_promises],
                )
            return promise

        def parse_statement(scope):
            token = peek()
            expr = parse_expression(scope)
            expect('\n')
            consume_all('\n')
            @Promise
            def promise():
                return IR.ExpressionStatement(
                    token,
                    expr.resolve(),
                )
            return promise

        def parse_expression(scope):
            return parse_postfix(scope)

        def make_function_call_promise(scope, token, expr, args):
            @Promise
            def promise():
                f = expr.resolve()
                if (isinstance(f, IR.Name) and
                        isinstance(f.definition, IR.FunctionDefinition)):
                    with scope.push(token):
                        f.definition.check_args(scope=scope, args=args)
                    return IR.SimpleFunctionCall(
                        token, f.definition.return_type, f.name, args,
                    )
                assert False, 'TODO'
            return promise

        def parse_postfix(scope):
            expr = parse_primary(scope)
            while True:
                if at('('):
                    args = expect_args()
                    expr = make_function_call_promise(
                        scope,
                        token,
                        expr,
                        args,
                    )

        def parse_global():
            token = peek()
            extern = bool(consume('extern'))
            if consume('struct'):
                return parse_struct(token=token, extern=extern)
            type_promise = parse_type()
            name = expect_id()
            if at('('):
                return parse_function(
                    token=token,
                    extern=extern,
                    type_promise=type_promise,
                    name=name,
                )
            else:
                return parse_global_variable(
                    token=token,
                    extern=extern,
                    type_promise=type_promise,
                    name=name,
                )
            with scope.push(token):
                raise scope.error('Expected struct or function definition')

        token = peek()
        import_promises = []
        definition_promises = []

        consume_all('\n')
        while at('import') or at('from'):
            import_promises.append(parse_import())
            consume_all('\n')

        while not at('EOF'):
            definition_promises.append(parse_global())
            consume_all('\n')

        @Promise
        def promise():
            imported = set()
            imports = []
            for p in import_promises:
                imp = p.resolve()
                if imp.name not in imported:
                    imported.add(imp.name)
                    imports.append(imp)
            definitions = [p.resolve() for p in definition_promises]
            return IR.TranslationUnit(
                token,
                imports,
                definitions,
            )

        return promise


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('file')
    argparser.add_argument('--search-dir', default='srcs')
    args = argparser.parse_args()
    ast = parser.parse(
        {
            'main': Source.from_path(args.file),
        },
        search_dir=args.search_dir,
    )
    print(ast)


if __name__ == '__main__':
    main()

