"""KC -> close interop with C.
"""
import argparse
import contextlib
import itertools
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


class FractalStringBuilder(object):
    def __init__(self, depth=0):
        self.parts = []
        self.depth = depth

    def __str__(self):
        parts = []
        self._dump(parts)
        return ''.join(parts)

    def _dump(self, parts):
        for part in self.parts:
            if isinstance(part, FractalStringBuilder):
                part._dump(parts)
            else:
                parts.append(str(part))

    def __iadd__(self, line):
        if '\n' in line:
            raise TypeError()
        # Ignore empty lines
        if line:
            self('  ' * self.depth + line + '\n')
        return self

    def __call__(self, s):
        self.parts.append(s)
        return self

    def spawn(self, depth_diff=0):
        child = FractalStringBuilder(self.depth + depth_diff)
        self.parts.append(child)
        return child


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
        cn = 0
        i = self.i
        s = self.source.data
        while i in range(len(s)) and s[i] != '\n':
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
            f'  In {self.source.filename} on line {self.lineno}\n'
            f'  {self.line}\n'
            f'  {" " * (self.colno - 1)}*\n'
        )


class Error(Exception):
    def __init__(self, tokens, message):
        super().__init__(f'{message}\n{"".join(t.format() for t in tokens)}')
        self.tokens = tuple(tokens)
        self.message = message


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
        if isinstance(value, (type(None), int, float, bool, str, CIR.CType)):
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
def CIR(ns):
    "C Intermediate Representation"

    # These are the keywords that a primitive type can be composed of
    PRIMITIVE_TYPE_SPECIFIERS = (
        'void',
        'char',
        'short',
        'int',
        'long',
        'unsigned',
        'float',
        'double',
    )
    ns(PRIMITIVE_TYPE_SPECIFIERS, 'PRIMITIVE_TYPE_SPECIFIERS')

    PRIMITIVE_TYPE_NAMES = PRIMITIVE_TYPE_SPECIFIERS + (
        # 'long long',  # (since C99)
        'unsigned char',
        'unsigned short',
        'unsigned int',
        'unsigned long',
        # 'unsigned long long',  # (since C99)
        'long double',
    )
    ns(PRIMITIVE_TYPE_NAMES, 'PRIMITIVE_TYPE_NAMES')

    C_KEYWORDS = {
        'auto',
        'break',
        'case',
        'char',
        'const',
        'continue',
        'default',
        'do',
        'double',
        'else',
        'enum',
        'extern',
        'float',
        'for',
        'goto',
        'if',
        'inline',  #  (since C99)
        'int',
        'long',
        'register',
        'restrict',  #  (since C99)
        'return',
        'short',
        'signed',
        'sizeof',
        'static',
        'struct',
        'switch',
        'typedef',
        'union',
        'unsigned',
        'void',
        'volatile',
        'while',
    }
    ns(C_KEYWORDS, 'C_KEYWORDS')

    OBJC_KEYWORDS = {'id'}
    ns(OBJC_KEYWORDS, 'OBJC_KEYWORDS')

    @ns
    def judge_c_name(name, tokens):
        if name.startswith('_'):
            return f'C names starting with underscore are reserved ({name})'
        if '__' in name:
            return (
                f'C names containing consecutive underscores are reserved '
                f'({name})'
            )
        if name in C_KEYWORDS:
            return f'{name} is a C keyword'
        if name in OBJC_KEYWORDS:
            return f'{name} is an Objective-C keyword'

    @ns
    def check_good_c_name(name, tokens):
        message = judge_c_name(name)
        if message is not None:
            raise Error(tokens, message)

    @ns
    class Id:
        "Proxy for str -> indicates value is used as a C identifier"
        def __init__(self, value):
            self.value = value

        def __repr__(self):
            return f'Id({repr(self.value)})'

        def __str__(self):
            return str(self.value)

        def __eq__(self, other):
            return type(self) is type(other) and self.value == other.value

        def __hash__(self):
            return hash((type(self), self.value))

    @ns
    class ScopeValue:
        """This mixin type indicates which classes are valid
        as values for parser.Scope. The keys of parser.Scope
        are always str.
        """

    @ns
    class ScopeVariableValue(ScopeValue):
        """If the scope returns an instance of this type,
        it means the name may be used like a variable.

        Subclasses should implement:

            name_type: CType
        """

    @ns
    class CType:
        def convertible_to(self, other):
            return self == other

    @ns
    class StructType(CType, ScopeValue):
        def __init__(self, token, name):
            self.token = token  # location where first encountered
            self.name = name

            # to be filled in at definition
            # (as opposed to declaration)
            self.token_at_definition = None
            self.fields_by_name = None  # dict: name -> StructField

        def __repr__(self):
            return f'StructType({self.name})'

        def __eq__(self, other):
            return type(self) is type(other) and self.name == other.name

        def __hash__(self):
            return hash((type(self), self.name))

    @ns
    class PrimitiveType(CType, ScopeValue):
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return f'PrimitiveType({self.name})'

        def __eq__(self, other):
            return type(self) is type(other) and self.name == other.name

        def __hash__(self):
            return hash((type(self), self.name))

        def convertible_to(self, other):
            return self == other or (
                type(self) is type(other) and
                (self.name, other.name) in {
                    ('int', 'long'),
                    ('int', 'double'),
                    ('float', 'double'),
                }
            )

    @ns
    class PointerType(CType):
        def __init__(self, base):
            self.base = base

        def __repr__(self):
            return f'PointerType({self.base})'

        def __eq__(self, other):
            return type(self) is type(other) and self.base == other.base

        def __hash__(self):
            return hash((type(self), self.base))

        def convertible_to(self, other):
            return self == other or other == PointerType(PrimitiveType('void'))

    @ns
    class FunctionType(CType):
        def __init__(self, return_type, parameters, vararg: bool):
            self.return_type = return_type
            self.parameters = parameters
            self.vararg = vararg

        def __repr__(self):
            return (
                f'FunctionType({self.return_type}, '
                f'{self.parameters}, {self.vararg})'
            )

        def __eq__(self, other):
            return type(self) == type(other) and [
                self.return_type, self.parameters,
            ] == [
                other.return_type, other.parameters,
            ]

        def __hash__(self):
            return hash((type(self), self.return_type, self.parameters))

    @ns
    class FunctionStub(ScopeVariableValue):
        def __init__(self, token, return_type, name, parameters, vararg):
            self.token = token
            self.return_type = return_type
            self.name = name
            self.parameters = parameters
            self.vararg = vararg

            # to be filled in at definition
            self.token_at_definition = None

        @property
        def name_type(self):
            return PointerType(self.type)

        @property
        def type(self):
            return FunctionType(
                self.return_type,
                [p.type for p in self.parameters],
                self.vararg)

        @property
        def signature(self):
            return (
                self.return_type,
                self.name,
                tuple(p.type for p in self.parameters),
                self.vararg)

        def matches(self, other: 'FunctionStub'):
            return self.signature == other.signature

    @ns
    class N(Node):
        pass

    @ns
    class GlobalDefinition(N):
        pass

    @ns
    class Statement(N):
        pass

    @ns
    class Expression(N):
        """
        All subclasses should implement:

            expression_type: CType
        """

    @ns
    class Block(Statement):
        fields = (
            ('statements', typeutil.List[Statement]),
        )

    @ns
    class Include(N):
        fields = (
            ('use_quotes', bool),  # quotes or angle brackets
            ('value', str),
        )

    @ns
    class TranslationUnit(N):
        fields = (
            ('includes', typeutil.List[Include]),
            ('definitions', typeutil.List[GlobalDefinition]),
        )

    @ns
    class StructField(N):
        fields = (
            ('type', CType),
            ('name', Id),
        )

    @ns
    class StructDefinition(GlobalDefinition):
        fields = (
            ('name', Id),
            ('fields', typeutil.List[StructField]),
        )

    @ns
    class Parameter(N):
        fields = (
            ('type', CType),
            ('name', Id),
        )

    @ns
    class FunctionDefinition(GlobalDefinition):
        fields = (
            ('return_type', CType),
            ('name', Id),
            ('parameters', typeutil.List[Parameter]),
            ('vararg', bool),
            ('body', Block),
        )

        @property
        def stub(self):
            return FunctionStub(
                self.token,
                self.return_type,
                self.name,
                self.parameters,
                self.vararg,
            )

        @property
        def type(self):
            return self.stub.type

    @ns
    class Return(Statement):
        fields = (
            ('expression', typeutil.Optional[Expression]),
        )

    @ns
    class ExpressionStatement(Statement):
        fields = (
            ('expression', Expression),
        )

    @ns
    class Name(Expression):
        fields = (
            ('definition', ScopeVariableValue),  # where name is defined,
            ('name', Id),
        )

        @property
        def expression_type(self):
            return self.definition.name_type

    @ns
    class IntLiteral(Expression):
        expression_type = PrimitiveType('int')

        fields = (
            ('value', int),
        )

    @ns
    class DoubleLiteral(Expression):
        expression_type = PrimitiveType('double')

        fields = (
            ('value', float),
        )

    @ns
    class StringLiteral(Expression):
        expression_type = PointerType(PrimitiveType('char'))

        fields = (
            ('value', str),
        )

    @ns
    class FunctionCall(Expression):
        fields = (
            ('expression_type', CType),
            ('f', Expression),
            ('args', typeutil.List[Expression]),
        )

    @ns
    class Operation(Expression):
        # This one should not have its constructor called directly...
        # instead construct through mkop
        fields = (
            ('name', str),
            ('args', typeutil.List[Expression]),
        )

        type_map = {
            ('+', PrimitiveType('int'), PrimitiveType('int')):
                PrimitiveType('int'),
            ('+', PrimitiveType('double'), PrimitiveType('int')):
                PrimitiveType('double'),
            ('+', PrimitiveType('int'), PrimitiveType('double')):
                PrimitiveType('double'),
            ('+', PrimitiveType('double'), PrimitiveType('double')):
                PrimitiveType('double'),
            ('-', PrimitiveType('int'), PrimitiveType('int')):
                PrimitiveType('int'),
        }

    @ns
    def mkop(stack, token, name, args):
        self = Operation(token, name, args)
        argtypes = [arg.expression_type for arg in self.args]
        key = (self.name,) + tuple(argtypes)
        if key not in type(self).type_map:
            raise Error(
                stack + [self.token],
                f'Operation {key} not supported')
        self.expression_type = type(self).type_map[key]
        return self


@Namespace
def lexer(ns):
    KEYWORDS = {
      'is', 'not', 'null', 'true', 'false', 'new', 'and', 'or', 'in',
      'inline', 'extern', 'class', 'trait', 'final', 'def', 'auto',
      'for', 'if', 'else', 'while', 'break', 'continue', 'return',
      'with', 'from', 'import', 'as', 'try', 'catch', 'finally', 'raise',
      'except', 'case','switch'
    } | set(CIR.C_KEYWORDS)
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
            while s.startswith('//', i) or i < len(s) and s[i] in ' \t\r\n':
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
def parser(ns):

    @ns
    class Scope:
        def __init__(self, parent: 'Scope', stack=None):
            assert stack is not None or parent is not None

            if stack is None:
                stack = parent.stack

            self.parent = parent
            self.table = dict()
            self.stack = stack
            self.included = set() if parent is None else parent.included
            self.cached_translation_units = (
                dict() if parent is None else parent.cached_translation_units
            )
            self.current_function: FunctionStub = (
                None if parent is None else parent.current_function
            )

        def __getitem__(self, key: str) -> CIR.ScopeValue:
            if key in self.table:
                return self.table[key]
            elif self.parent is not None:
                return self.parent[key]
            else:
                raise Error(self.stack, f'Name {key} is not declared')

        def __setitem__(self, key: str, value: CIR.ScopeValue):
            if key in self.table:
                raise Error(
                    self.stack + [self.table[key].token, value.token],
                    f'{key} already defined in this scope')
            self.table[key] = value

        def __contains__(self, key: str):
            return (
                key in self.table or
                self.parent is not None and key in self.parent
            )

        @property
        def struct_names(self):
            return StructNamesContainer(self)

    class StructNamesContainer:
        def __init__(self, scope):
            self.scope = scope

        def __contains__(self, name):
            return isinstance(scope[name], CIR.StructType)
    @ns
    def new_global_scope():
        scope = Scope(None, [])
        for name in CIR.PRIMITIVE_TYPE_NAMES:
            scope[name] = CIR.PrimitiveType(name)
        return scope

    @ns
    def parse(source: Source, scope: Scope):
        stack = scope.stack
        tokens = lexer.lex(source)
        i = 0

        @contextlib.contextmanager
        def push(token: Token):
            stack.append(token)
            try:
                yield
            finally:
                stack.pop()

        def error(message: str):
            return Error(stack, message)

        def peek():
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

        def parse_id():
            return CIR.Id(expect('NAME').value)

        def at_type():
            token = peek()
            return (
                token.type in CIR.PRIMITIVE_TYPE_SPECIFIERS or
                token.type == 'NAME' and token.name in scope.struct_names
            )

        def parse_type():
            token = peek()
            if token.type in CIR.PRIMITIVE_TYPE_SPECIFIERS:
                parts = [gettok().type]
                while peek().type in CIR.PRIMITIVE_TYPE_SPECIFIERS:
                    parts.append(gettok().type)
                name = ' '.join(parts)
                with push(token):
                    t = scope[name]
                assert isinstance(t, CIR.PrimitiveType), t
            elif consume('NAME'):
                with push(peek()):
                    t = scope[token.name]
                    if not isinstance(t, CIR.StructType):
                        with push(t.token):
                            raise error(f'{token.name} is not a type name')
            else:
                with push(token):
                    raise error(f'Expected type but got {token}')
            while consume('*'):
                t = CIR.PointerType(t)
            return t

        def parse_global(out):
            # this function accepts an 'out' argument because
            # parsing a global definition doesn't always result in
            # a concrete GlobalDefinition Node.
            # E.g. forward declarations update the scope, but
            # doesn't actually result in a GlobalDefinition.
            token = peek()
            extern = consume('extern')
            if at('struct'):
                parse_struct(out=out, token=token, extern=extern)
            elif at_type():
                type = parse_type()
                name = parse_id()
                if at('('):
                    parse_function(
                        out=out,
                        token=token,
                        extern=extern,
                        type=type,
                        name=name,
                    )
                else:
                    parse_global_variable(
                        out=out,
                        token=token,
                        extern=extern,
                        type=type,
                        name=name,
                    )
                return
            with push(token):
                raise error(
                    f'Expected struct, function or variable definition')

        def declare_function(stub):
            if stub.name not in scope:
                scope[stub.name.value] = stub
                return

            oldstub = scope[stub.name]
            if type(oldstub) != CIR.FunctionStub or not oldstub.matches(stub):
                with push(stub.token), push(oldstub.token):
                    raise error(
                        f'Declarations for {stub.name} does not match')

        def parse_function(out, token, extern, type, name):
            params, vararg = parse_params()
            stub = CIR.FunctionStub(token, type, name, params, vararg)
            declare_function(stub)

            if extern:
                with push(token):
                    # All functions are going to be 'extern' by
                    # normal C standards.
                    raise error(
                        'Semantics for "extern" functions not defined yet')

            if consume(';'):
                # If this is where it ends, this was just a declaration
                # and not a definition
                return

            function_scope = Scope(scope)
            function_scope.current_function = stub
            for param in params:
                function_scope[param.name] = name

            body = parse_block(function_scope)

            if (type != CIR.PrimitiveType('void') and
                    not analyzer.returns(body)):
                with push(token):
                    raise error('Function may not return')

            out.append(CIR.FunctionDefinition(
                token,
                type,
                name,
                params,
                vararg,
                body,
            ))

        def parse_params():
            params = []
            vararg = False
            expect('(')
            while not consume(')'):
                if consume('...'):
                    vararg = True
                    expect(')')
                    break
                token = peek()
                type = parse_type()
                name = parse_id()
                params.append(CIR.Parameter(token, type, name))
                if not consume(','):
                    expect(')')
                    break
            return params, vararg

        def parse_block(parent_scope):
            scope = Scope(parent_scope)
            token = peek()
            statements = []
            expect('{')
            while not consume('}'):
                statements.append(parse_statement(scope))
            return CIR.Block(token, statements)

        def parse_statement(scope):
            token = peek()
            if at('{'):
                return parse_block(scope)
            if consume('return'):
                if consume(';'):
                    expr = None
                else:
                    expr = parse_expression(scope)

                tp = (
                    CIR.PrimitiveType('void')
                        if expr is None else expr.expression_type
                )
                cf = scope.current_function
                if cf is None:
                    with push(token):
                        raise error('Tried to return outside function')

                if not tp.convertible_to(cf.return_type):
                    with push(token), push(cf.token):
                        raise error(
                            f'Tried to return {tp}, but expected '
                            f'{cf.return_type}')

                if expr is not None and tp == CIR.PrimitiveType('void'):
                    with push(token):
                        raise error('You cannot return a void expression')

                expect(';')
                return CIR.Return(token, expr)
            expr = parse_expression(scope)
            expect(';')
            return CIR.ExpressionStatement(token, expr)

        def parse_expression(scope):
            return parse_additive(scope)

        def parse_args(scope):
            expect('(')
            args = []
            while not consume(')'):
                args.append(parse_expression(scope))
                if not consume(','):
                    expect(')')
                    break
            return args

        def parse_additive(scope):
            expr = parse_multiplicative(scope)
            while True:
                token = peek()
                if consume('+'):
                    args = [expr, parse_multiplicative(scope)]
                    expr = CIR.mkop(stack, token, '+', args)
                elif consume('-'):
                    args = [expr, parse_multiplicative(scope)]
                    expr = CIR.mkop(stack, token, '-', args)
                else:
                    break
            return expr

        def parse_multiplicative(scope):
            expr = parse_unary(scope)
            while True:
                token = peek()
                if consume('*'):
                    args = [expr, parse_unary(scope)]
                    expr = CIR.mkop(stack, token, '*', args)
                elif consume('/'):
                    args = [expr, parse_unary(scope)]
                    expr = CIR.mkop(stack, token, '/', args)
                elif consume('%'):
                    args = [expr, parse_unary(scope)]
                    expr = CIR.mkop(stack, token, '%', args)
                else:
                    break
            return expr

        def parse_unary(scope):
            return parse_postfix(scope)

        def parse_postfix(scope):
            expr = parse_primary(scope)
            while True:
                token = peek()
                if at('('):
                    args = parse_args(scope)
                    et = expr.expression_type
                    return_type = check_func_args(token, expr, args)
                    expr = CIR.FunctionCall(
                        token,
                        return_type,
                        expr,
                        args,
                    )
                    continue
                break
            return expr

        def check_func_args(token, fexpr, args):
            # check all args have 'expression_type'
            for arg in args:
                arg.expression_type

            pft = fexpr.expression_type
            with push(token):
                if not isinstance(pft, CIR.PointerType):
                    raise error('Not a function')
                ft = pft.base
                if not isinstance(ft, CIR.FunctionType):
                    raise error('Not a function')

                if ft.vararg:
                    if len(ft.parameters) > len(args):
                        raise error(
                            f'Expected at least {len(ft.parameters)} '
                            f'args but got {len(args)}')
                else:
                    if len(ft.parameters) != len(args):
                        raise error(
                            f'Expected {len(ft.parameters)} '
                            f'but got {len(args)}')
            for param, arg in zip(ft.parameters, args):
                if not arg.expression_type.convertible_to(param):
                    with push(arg.token):
                        raise error(
                            f'Expected argument of type {param.type} '
                            f'but got {arg.expression_type}')
            return ft.return_type

        def parse_primary(scope):
            token = peek()
            if consume('INT'):
                return CIR.IntLiteral(token, token.value)
            if consume('FLOAT'):
                return CIR.DoubleLiteral(token, token.value)
            if at('NAME'):
                id = parse_id()
                with push(token):
                    defn = scope[id.value]

                if not isinstance(defn, CIR.ScopeVariableValue):
                    with push(token), push(defn):
                        raise error(f'{id.value} is not a variable')

                return CIR.Name(token, defn, id)

            elif consume('STRING'):
                value = token.value
                return CIR.StringLiteral(token, value)

            with push(token):
                raise error(f'Expected expression but got {peek()}')

        def parse_include():
            token = expect('#')
            if expect('NAME').value != 'include':
                with push(token):
                    raise error('Expected "include"')

            if consume('<'):
                use_quotes = False
                parts = []
                while not consume('>'):
                    parts.append(gettok().value)
                name = ''.join(parts)
            else:
                use_quotes = True
                name = expect('STRING').value

            if not name.endswith('.kc'):
                return CIR.Include(token, use_quotes, name)
            else:
                header_name = header_name_from_kc_path(name)
                path = os.path.abspath(os.path.join(
                    os.path.dirname(source.filename),
                    name,
                ))
                if not path in scope.included:
                    scope.included.add(path)
                    new_source = Source.from_path(path)
                    with push(token):
                        tu = parser.parse(new_source, scope)
                    assert path not in scope.cached_translation_units, path
                    scope.cached_translation_units[path] = tu
                return CIR.Include(token, use_quotes, header_name)

        def parse_translation_unit():
            token = peek()
            includes = []
            defs = []
            consume_all(';')
            while at('#'):
                includes.append(parse_include())
                consume_all(';')
            while not at('EOF'):
                parse_global(defs)
                consume_all(';')
            return CIR.TranslationUnit(token, includes, defs)

        return parse_translation_unit()

    @ns
    def header_name_from_kc_path(path):
        assert path.endswith('.kc'), path
        name = os.path.basename(path)
        return name[:-len('.kc')] + '.h'

    @ns
    def source_name_from_kc_path(path):
        assert path.endswith('.kc'), path
        name = os.path.basename(path)
        return name[:-len('.kc')] + '.c'


@Namespace
def analyzer(ns):

    returns = Multimethod('returns')
    ns(returns, 'returns')

    @returns.on(CIR.Block)
    def returns(self):
        return any(map(returns, self.statements))

    @returns.on(CIR.Return)
    def returns(self):
        return True

    @returns.on(CIR.ExpressionStatement)
    def returns(self):
        return False


@Namespace
def C(ns):

    @ns
    def write_out(tu: CIR.TranslationUnit, outdir):
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
        basename = os.path.basename(tu.token.source.filename)
        assert basename.endswith('.kc'), basename
        header_name = parser.header_name_from_kc_path(basename)
        header_content = header_for(tu)
        with open(os.path.join(outdir, header_name), 'w') as f:
            f.write(header_content)
        source_name = parser.source_name_from_kc_path(basename)
        source_content = source_for(tu)
        with open(os.path.join(outdir, source_name), 'w') as f:
            f.write(source_content)

    @ns
    def header_for(tu: CIR.TranslationUnit):
        out = FractalStringBuilder()
        basename = os.path.basename(tu.token.source.filename)
        header_name = parser.header_name_from_kc_path(basename)
        header_macro = header_name.replace('.', '_').replace('/', '_')

        out += f'#ifndef {header_macro}'
        out += f'#define {header_macro}'

        for inc in tu.includes:
            if inc.use_quotes:
                out += f'#include "{inc.value}"'
            else:
                out += f'#include <{inc.value}>'

        for defn in tu.definitions:
            if isinstance(defn, CIR.FunctionDefinition):
                out += f'{proto_for(defn)};'

        out += f'#endif/*{header_macro}*/'

        return str(out)

    @ns
    def source_for(tu: CIR.TranslationUnit):
        out = FractalStringBuilder()
        basename = os.path.basename(tu.token.source.filename)
        header_name = parser.header_name_from_kc_path(basename)

        out += f'#include "{header_name}"'

        for defn in tu.definitions:
            translate(defn, out)

        return str(out)

    proto_for = Multimethod('proto_for')

    @proto_for.on(CIR.FunctionDefinition)
    def proto_for(fd):
        return declare(
            fd.type, translate(fd.name), [p.name for p in fd.parameters])

    declare = Multimethod('declare')

    @declare.on(CIR.FunctionType)
    def declare(ft, name: str, argnames=None):
        # The argument list can't bind tighter than anything in the front
        # that's already there. However, if name represents a pointer
        # to something, we need parentheses there, since the argument
        # list will bind tighter than the '*' which would change the meaning.
        if name and not name[0].isalnum():
            name = f'({name})'

        if argnames is None:
            argnames = [''] * len(ft.parameters)

        params = ', '.join([
            declare(pt, argname)
            for argname, pt in zip(argnames, ft.parameters)
        ] + (
            ['...'] if ft.vararg else []
        ))
        return declare(ft.return_type, name + f'({params})')

    @declare.on(CIR.PointerType)
    def declare(pt, name: str):
        return declare(pt.base, '*' + name)

    @declare.on(CIR.PrimitiveType)
    def declare(pt, name: str):
        return f'{pt.name} {name}'

    @declare.on(CIR.StructType)
    def declare(st, name: str):
        return f'{st.name} {name}'

    translate = Multimethod('translate')

    @translate.on(CIR.Id)
    def translate(id_):
        return id_.value

    @translate.on(CIR.PointerType)
    def translate(pt):
        return f'{translate(pt.base)}*'

    @translate.on(CIR.PrimitiveType)
    def translate(pt):
        return pt.name

    @translate.on(CIR.StructType)
    def translate(st):
        return st.name

    @translate.on(CIR.Parameter)
    def translate(param):
        return f'{translate(param.type)} {translate(param.name)}'

    @translate.on(CIR.FunctionDefinition)
    def translate(self, out):
        out += proto_for(self)
        translate(self.body, out)

    @translate.on(CIR.Block)
    def translate(self, out):
        out += '{'
        inner = out.spawn(1)
        out += '}'

        for stmt in self.statements:
            translate(stmt, inner)

    @translate.on(CIR.Return)
    def translate(self, out):
        if self.expression is None:
            out += 'return;'
        else:
            out += f'return {translate(self.expression)};'

    @translate.on(CIR.ExpressionStatement)
    def translate(self, out):
        out += translate(self.expression) + ';'

    @translate.on(CIR.FunctionCall)
    def translate(self):
        f = translate(self.f)
        if not isinstance(self.f, CIR.Name):
            f = f'({f})'
        args = ', '.join(map(translate, self.args))
        return f'{f}({args})'

    @translate.on(CIR.Name)
    def translate(self):
        return self.name.value

    @translate.on(CIR.IntLiteral)
    def translate(self):
        return str(self.value)

    @translate.on(CIR.DoubleLiteral)
    def translate(self):
        return str(self.value)

    @translate.on(CIR.StringLiteral)
    def translate(self):
        s = (self.value
            .replace('\\', '\\\\')
            .replace('\t', '\\t')
            .replace('\n', '\\n')
            .replace('\r', '\\r')
            .replace('"', '\\"')
            .replace("'", "\\'"))
        return f'"{s}"'

    @translate.on(CIR.Operation)
    def translate(self):
        if self.name == '+':
            left, right = map(translate, self.args)
            return f'({left} + {right})'
        elif self.name == '-':
            left, right = map(translate, self.args)
            return f'({left} - {right})'
        elif self.name == '*':
            left, right = map(translate, self.args)
            return f'({left} * {right})'
        elif self.name == '/':
            left, right = map(translate, self.args)
            return f'({left} / {right})'
        elif self.name == '%':
            left, right = map(translate, self.args)
            return f'({left} % {right})'
        else:
            with push(self.token):
                raise error(f'Unrecognized operation {self.name}')


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('path')
    argparser.add_argument('--out-dir', default='out')
    args = argparser.parse_args()
    scope = parser.new_global_scope()
    source = Source.from_path(args.path)
    main_translation_unit = parser.parse(source=source, scope=scope)
    C.write_out(main_translation_unit, outdir=args.out_dir)
    for tu in scope.cached_translation_units.values():
        C.write_out(tu, outdir=args.out_dir)

if __name__ == '__main__':
    main()
