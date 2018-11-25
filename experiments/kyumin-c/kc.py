"""KC -> close interop with C.
"""
import argparse
import contextlib
import itertools
import os
import shutil
import typing

_scriptdir = os.path.dirname(os.path.realpath(__file__))

UNIVERSAL_PREFIX = 'KLC'
ENCODED_NAME_PREFIX = UNIVERSAL_PREFIX + 'N'

MAIN_MODULE_NAME = 'main'


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
def IR(ns):
    "Intermediate Representation"

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
        'size_t',
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

    class IdMeta:
        def __instancecheck__(self, instance):
            return isinstance(instance, str)

        def __call__(self, s):
            if not isinstance(s, self):
                raise TypeError(f'{repr(s)} is not an Id')
            return s

    # Marker for str used as identifiers
    Id = IdMeta()
    ns(Id, 'Id')

    @ns
    class ScopeValue:
        """This mixin type indicates which classes are valid
        as values for parser.Scope. The keys of parser.Scope
        are always str.

        Subclasses should implement

            name: Id
        """

    @ns
    class ScopeVariableValue(ScopeValue):
        """If the scope returns an instance of this type,
        it means the name may be used like a variable.

        Subclasses should implement:

            name_type: CType
        """

    @ns
    class ScopeTypeValue(ScopeValue):
        """If the scope returns an instance of this type,
        it means that the name may be used like a type.
        """

    @ns
    class CType:
        pass

    @ns
    class StructType(CType, ScopeTypeValue):
        token_at_definition = None
        fields_by_name = None  # dict: name -> StructField

        def __init__(self, token, name):
            self.token = token  # location where first encountered
            self.name = name

        def __repr__(self):
            return f'StructType({self.name})'

        def __eq__(self, other):
            return type(self) is type(other) and self.name == other.name

        def __hash__(self):
            return hash((type(self), self.name))

        def matches(self, other):
            return type(self) is type(other) and self.name == other.name

    @ns
    class VarType(CType, ScopeTypeValue):
        def __repr__(self):
            return 'VarType()'

        def __eq__(self, other):
            return type(self) is type(other)

        def __hash__(self):
            return hash(type(self))

    @ns
    class PrimitiveType(CType, ScopeTypeValue):
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return f'PrimitiveType({self.name})'

        def __eq__(self, other):
            return type(self) is type(other) and self.name == other.name

        def __hash__(self):
            return hash((type(self), self.name))


    INTEGRAL_TYPES = tuple(
        PrimitiveType(t) for t in PRIMITIVE_TYPE_NAMES if t != 'void')
    ns(INTEGRAL_TYPES, 'INTEGRAL_TYPES')

    @ns
    class ConstType(CType):
        def __init__(self, base):
            self.base = base

        def __repr__(self):
            return f'ConstType({self.base})'

        def __eq__(self, other):
            return type(self) is type(other) and self.base == other.base

        def __hash__(self):
            return hash((type(self), self.base))

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

    # convertible(a, b) asks is a implicitly convertible to b
    convertible = Multimethod('convertible', 2)
    ns(convertible, 'convertible')

    @convertible.on(CType, ConstType)
    def convertible(a, b):
        return convertible(a, b.base)

    @convertible.on(ConstType, CType)
    def convertible(a, b):
        return convertible(a.base, b)

    @convertible.on(PointerType, PointerType)
    def convertible(a, b):
        return (
            a == b or
            a.base == b.base or
            b == PointerType(PrimitiveType('void')) or
            not isinstance(a.base, ConstType) and convertible(a.base, b.base)
        )

    @convertible.on(PointerType, CType)
    def convertible(a, b):
        return False

    @convertible.on(CType, PointerType)
    def convertible(a, b):
        return False

    @convertible.on(StructType, CType)
    def convertible(a, b):
        return a == b

    @convertible.on(CType, StructType)
    def convertible(a, b):
        return a == b

    @convertible.on(PrimitiveType, PrimitiveType)
    def convertible(a, b):
        return a == b or (
            (a.name, b.name) in (
                ('int', 'long'),
                ('int', 'double'),
                ('float', 'double'),
            )
        )

    @convertible.on(VarType, VarType)
    def convertible(a, b):
        return True

    @ns
    class ClassStub(ScopeVariableValue):
        token_at_definition = None
        fields_by_name = None

        def __init__(self, tokne, name):
            self.name = name

        def matches(self, other):
            return type(self) is type(other) and self.name == other.name

    @ns
    class FunctionStub(ScopeVariableValue):
        def __init__(
                self,
                token,
                extern,
                return_type,
                name,
                parameters,
                vararg):
            self.token = token
            self.extern = extern
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
                self.extern,
                self.return_type,
                self.name,
                tuple(p.type for p in self.parameters),
                self.vararg)

        def matches(self, other: 'FunctionStub'):
            return (
                type(self) is type(other) and
                self.signature == other.signature
            )

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
            ('name', str),
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
    class ClassDefinition(GlobalDefinition):
        fields = (
            ('name', Id),
        )

    @ns
    class StructDefinition(GlobalDefinition):
        fields = (
            ('extern', bool),
            ('name', Id),
            ('fields', typeutil.List[StructField]),
        )

    @ns
    class Parameter(N, ScopeVariableValue):
        fields = (
            ('type', CType),
            ('name', Id),
        )

        @property
        def name_type(self):
            return self.type

    @ns
    class FunctionDefinition(GlobalDefinition):
        fields = (
            ('extern', bool),
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
                self.extern,
                self.return_type,
                self.name,
                self.parameters,
                self.vararg,
            )

        @property
        def type(self):
            return self.stub.type

    @ns
    class LocalVariableDefinition(Statement, ScopeVariableValue):
        fields = (
            ('type', CType),
            ('name', Id),
            ('expression', typeutil.Optional[Expression]),
        )

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not isinstance(self.type, ConstType), self

        @property
        def name_type(self):
            return self.type

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
    class CStringLiteral(Expression):
        expression_type = PointerType(ConstType(PrimitiveType('char')))

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

        _type_list_cache = dict()

        type_table = {
            '+': [
                [
                    tuple(map(PrimitiveType, [
                        'int', 'int',
                    ])),
                    PrimitiveType('int'),
                ],
                [
                    tuple(map(PrimitiveType, [
                        'double', 'int',
                    ])),
                    PrimitiveType('double'),
                ],
                [
                    tuple(map(PrimitiveType, [
                        'int', 'double',
                    ])),
                    PrimitiveType('double'),
                ],
                [
                    tuple(map(PrimitiveType, [
                        'double', 'double',
                    ])),
                    PrimitiveType('double'),
                ],
            ],
            '-': [
                [
                    tuple(map(PrimitiveType, [
                        'int', 'int',
                    ])),
                    PrimitiveType('int'),
                ],
            ],
        }

    @ns
    def mkop(stack, token, name, args):
        self = Operation(token, name, args)
        argtypes = [arg.expression_type for arg in self.args]
        key = (self.name,) + tuple(argtypes)

        # Check types, and determine the type of the expression
        if key not in type(self)._type_list_cache:
            for expargtypes, rtype in type(self).type_table[self.name]:
                if len(expargtypes) == len(args) and all(
                        convertible(arg.expression_type, expargtype)
                        for expargtype, arg in zip(expargtypes, args)):
                    type(self)._type_list_cache[key] = rtype
                    break
            else:
                raise Error([token], f'Unsupported operation {key}')
        self.expression_type = type(self)._type_list_cache[key]

        return self


@Namespace
def lexer(ns):
    KEYWORDS = {
      'is', 'not', 'null', 'true', 'false', 'new', 'and', 'or', 'in',
      'inline', 'extern', 'class', 'trait', 'final', 'def', 'auto',
      'for', 'if', 'else', 'while', 'break', 'continue', 'return',
      'with', 'from', 'import', 'as', 'try', 'catch', 'finally', 'raise',
      'except', 'case','switch', 'var',
    } | set(IR.C_KEYWORDS)
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
def parser(ns):

    @ns
    class Scope:
        def __init__(self, parent: 'Scope', stack=None):
            assert stack is not None or parent is not None

            if stack is None:
                stack = parent.stack

            self.root = self if parent is None else parent.root
            self.parent = parent
            self.table = dict()
            self.stack = stack
            self.current_function: FunctionStub = (
                None if parent is None else parent.current_function
            )
            self.module_name: str = (
                None if parent is None else parent.module_name
            )
            self.source_root = (
                None if parent is None else parent.source_root
            )

            if self.root is self:
                self._module_scope_cache = dict()
                self._translation_unit_cache = dict()
                self._stub_map = dict()

        @property
        def stub_map(self):
            return self.root._stub_map

        def add_stub(self, stub):
            if stub.name in self.stub_map:
                oldstub = self.stub_map[stub.name]
                if not stub.matches(oldstub):
                    with push(stub.token), push(oldstub.token):
                        raise error('Mismatched declaration')
                return oldstub
            else:
                self.stub_map[stub.name] = stub
                return stub

        @property
        def module_scope_cache(self):
            return self.root._module_scope_cache

        @property
        def translation_unit_cache(self):
            return self.root._translation_unit_cache

        def __getitem__(self, key: str) -> IR.ScopeValue:
            if key in self.table:
                return self.table[key]
            elif self.parent is not None:
                return self.parent[key]
            else:
                raise Error(self.stack, f'Name {key} is not declared')

        def __setitem__(self, key: str, value: IR.ScopeValue):
            assert isinstance(value, IR.ScopeValue), value
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

        @contextlib.contextmanager
        def namespace(self, module_name):
            scope = Scope(self)
            scope.module_name = module_name
            yield scope

        @property
        def struct_names(self):
            return StructNamesContainer(self)

    class StructNamesContainer:
        def __init__(self, scope):
            self.scope = scope

        def __contains__(self, name):
            return isinstance(self.scope[name], IR.StructType)
    @ns
    def new_global_scope():
        scope = Scope(None, [])
        for name in IR.PRIMITIVE_TYPE_NAMES:
            scope[name] = IR.PrimitiveType(name)
        return scope

    @ns
    def parse(source: Source, scope: Scope):
        assert scope.module_name, (
            'Scopes passed to parser.parse must have a module name'
        )
        assert scope.parent is scope.root, (
            'Scopes passed to parser.parse must have exactly one ancestor'
        )
        stack = scope.stack
        tokens = lexer.lex(source)
        i = 0
        indent_stack = []

        @contextlib.contextmanager
        def push(token: Token):
            stack.append(token)
            try:
                yield
            finally:
                stack.pop()

        def error(message: str):
            return Error(stack, message)

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

        def parse_id():
            return IR.Id(expect('NAME').value)

        def at_type():
            token = peek()
            return (
                token.type == 'var' or
                token.type in IR.PRIMITIVE_TYPE_SPECIFIERS or
                token.type == 'NAME' and token.value in scope.struct_names
            )

        def parse_type():
            token = peek()
            if consume('var'):
                t = IR.VarType()
            elif token.type in IR.PRIMITIVE_TYPE_SPECIFIERS:
                parts = [gettok().type]
                while peek().type in IR.PRIMITIVE_TYPE_SPECIFIERS:
                    parts.append(gettok().type)
                name = ' '.join(parts)
                with push(token):
                    t = scope[name]
                assert isinstance(t, IR.PrimitiveType), t
            elif consume('NAME'):
                with push(token):
                    t = scope[token.value]
                    if not isinstance(t, IR.ScopeTypeValue):
                        with push(t.token):
                            raise error(f'{token.name} is not a type name')
            else:
                with push(token):
                    raise error(f'Expected type but got {token}')
            while True:
                if consume('*'):
                    t = IR.PointerType(t)
                elif consume('const'):
                    t = IR.ConstType(t)
                else:
                    break
            return t

        def qualify_name(name):
            return scope.module_name + '.' + name

        def parse_global(out):
            # this function accepts an 'out' argument because
            # parsing a global definition doesn't always result in
            # a concrete GlobalDefinition Node.
            # E.g. forward declarations update the scope, but
            # doesn't actually result in a GlobalDefinition.
            token = peek()
            extern = bool(consume('extern'))
            if consume('struct'):
                parse_struct(out=out, token=token, extern=extern)
                return
            elif consume('class'):
                parse_class(out=out, token=token, extern=extern)
                return
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
                    f'Expected struct, class, function or '
                    f' variable definition but got {token}')

        def parse_struct(out, token, extern):
            name = parse_id()
            qualified_name = name if extern else qualify_name(name)
            struct_type = scope.add_stub(
                IR.StructType(token, qualified_name))

            if name not in scope or scope[name] is not struct_type:
                scope[name] = struct_type

            if consume('\n'):
                # if there's newline here, it's just a declaration
                return

            if struct_type.token_at_definition is not None:
                with push(token), push(struct_type.token_at_definition):
                    raise error(
                        f'Duplicate definition for struct '
                        f'{qualified_name}')

            struct_type.token_at_definition = token
            assert struct_type.fields_by_name is None

            fields = []
            fields_by_name = dict()

            expect('{')
            consume_all('\n')
            while not consume('}'):
                field_token = peek()
                extern_field = consume('extern')
                if extern and extern_field:
                    with push(field_token):
                        raise error(
                            f'In an extern struct, '
                            f'all fields are already extern')
                field_type = parse_type()
                if isinstance(field_type, IR.StructType):
                    if field_type.token_at_definition is None:
                        with push(field_token):
                            raise error(
                                f'Structs cannot have fields '
                                f'with incomplete type')
                field_name = parse_id()
                qualified_field_name = (
                    field_name if extern or extern_field else
                    qualify_name(field_name)
                )
                field = IR.StructField(
                    field_token,
                    field_type,
                    qualified_field_name,
                )
                fields.append(field)
                fields_by_name[field_name] = field
                expect('\n')
                consume_all('\n')

            struct_type.fields_by_name = fields_by_name

            out.append(IR.StructDefinition(
                token, extern, qualified_name, fields))

        def parse_class(out, token, extern):
            name = parse_id()
            qualified_name = name if extern else qualify_name(name)
            stub = scope.add_stub(IR.ClassStub(token, qualified_name))

            if name not in scope or scope[name] is not stub:
                scope[name] = stub

            if consume('\n'):
                return

            if stub.token_at_definition is not None:
                with push(token), push(stub.token_at_definition):
                    raise error(
                        f'Duplicate definition for class '
                        f'{qualified_name}')

            stub.token_at_definition = token
            assert stub.fields_by_name is None

            fields = []
            fields_by_name = dict()

            expect('{')
            consume_all('\n')
            expect('}')

            out.append(IR.ClassDefinition(token, qualified_name))

        def declare_function(name, function_stub_args):
            stub = scope.add_stub(IR.FunctionStub(*function_stub_args))

            if name not in scope or scope[name] is not stub:
                scope[name] = stub

            return stub

        def parse_function(out, token, extern, type, name):
            params, vararg = parse_params()
            qualified_name = name if extern else qualify_name(name)
            stub = declare_function(
                name, [token, extern, type, qualified_name, params, vararg])

            if consume('\n'):
                # If this is where it ends, this was just a declaration
                # and not a definition
                return

            if stub.token_at_definition is not None:
                with push(token), push(stub.token_at_definition):
                    raise error(
                        f'Duplicate definition for '
                        f'function {qualified_name}')

            stub.token_at_definition = token

            function_scope = Scope(scope)
            function_scope.current_function = stub
            for param in params:
                function_scope[param.name] = param

            body = parse_block(function_scope)

            if (type != IR.PrimitiveType('void') and
                    not analyzer.returns(body)):
                with push(token):
                    raise error('Control reaches end of non-void function')

            # Make sure that even void functions always have
            # an explicit return.
            # This makes sure that the release pool always gets
            # cleaned up.
            if (type == IR.PrimitiveType('void') and
                    not analyzer.returns(body)):
                body.statements.append(IR.Return(token, None))

            assert analyzer.returns(body)

            out.append(IR.FunctionDefinition(
                token,
                extern,
                type,
                qualified_name,
                params,
                vararg,
                body,
            ))

        def parse_params():
            params = []
            vararg = False
            expect('(')
            with skipping_newlines(True):
                while not consume(')'):
                    if consume('...'):
                        vararg = True
                        expect(')')
                        break
                    token = peek()
                    type = parse_type()
                    name = parse_id()
                    params.append(IR.Parameter(token, type, name))
                    if not consume(','):
                        expect(')')
                        break
                return params, vararg

        def parse_block(parent_scope):
            scope = Scope(parent_scope)
            token = peek()
            statements = []
            expect('{')
            consume_all('\n')

            # We need to validate that const variable definitions
            # are not mixed with non-variable definitions, otherwise
            # there's no good way to translate to C89 compatible code.
            only_decls = True

            while not consume('}'):
                stmt = parse_statement(scope)
                if (not only_decls and
                        isinstance(stmt, IR.LocalVariableDefinition) and
                        isinstance(stmt.type, IR.ConstType)):
                    with push(stmt.token):
                        raise error(
                            f'const variable definitions cannot be '
                            f'mixed with non-declarations')
                if (only_decls and
                        not isinstance(stmt, IR.LocalVariableDefinition)):
                    only_decls = False
                statements.append(stmt)

            consume_all('\n')
            return IR.Block(token, statements)

        def parse_statement(scope):
            token = peek()
            if at('{'):
                return parse_block(scope)
            if at_type():
                type = parse_type()
                name = parse_id()
                expr = parse_expression(scope) if consume('=') else None
                expect('\n')
                if (expr is not None and
                        not IR.convertible(expr.expression_type, type)):
                    with push(token):
                        raise error(
                            f'Tried to set value of type '
                            f'{expr.expression_type} to variable of '
                            f'type {type}')
                if isinstance(type, IR.ConstType):
                    with push(token):
                        raise error(
                            f'Declaring const variables is not supported')
                defn = IR.LocalVariableDefinition(
                    token,
                    type,
                    name,
                    expr,
                )
                scope[name] = defn
                return defn
            if consume('return'):
                if consume('\n'):
                    expr = None
                else:
                    expr = parse_expression(scope)

                tp = (
                    IR.PrimitiveType('void')
                        if expr is None else expr.expression_type
                )
                cf = scope.current_function
                if cf is None:
                    with push(token):
                        raise error('Tried to return outside function')

                if not IR.convertible(tp, cf.return_type):
                    with push(token), push(cf.token):
                        raise error(
                            f'Tried to return {tp}, but expected '
                            f'{cf.return_type}')

                if expr is not None and tp == IR.PrimitiveType('void'):
                    with push(token):
                        raise error('You cannot return a void expression')

                expect('\n')
                return IR.Return(token, expr)
            expr = parse_expression(scope)
            expect('\n')
            return IR.ExpressionStatement(token, expr)

        def parse_expression(scope):
            return parse_additive(scope)

        def parse_args(scope):
            expect('(')
            with skipping_newlines(True):
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
                    expr = IR.mkop(stack, token, '+', args)
                elif consume('-'):
                    args = [expr, parse_multiplicative(scope)]
                    expr = IR.mkop(stack, token, '-', args)
                else:
                    break
            return expr

        def parse_multiplicative(scope):
            expr = parse_unary(scope)
            while True:
                token = peek()
                if consume('*'):
                    args = [expr, parse_unary(scope)]
                    expr = IR.mkop(stack, token, '*', args)
                elif consume('/'):
                    args = [expr, parse_unary(scope)]
                    expr = IR.mkop(stack, token, '/', args)
                elif consume('%'):
                    args = [expr, parse_unary(scope)]
                    expr = IR.mkop(stack, token, '%', args)
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
                    expr = IR.FunctionCall(
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
                if not isinstance(pft, IR.PointerType):
                    raise error('Not a function')
                ft = pft.base
                if not isinstance(ft, IR.FunctionType):
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
                if not IR.convertible(arg.expression_type, param):
                    with push(arg.token):
                        raise error(
                            f'Expected argument of type {param} '
                            f'but got {arg.expression_type}')
            return ft.return_type

        def parse_primary(scope):
            token = peek()
            if consume('('):
                with skipping_newlines(True):
                    expr = parse_expression()
                    expect(')')
                    return expr
            if consume('INT'):
                return IR.IntLiteral(token, token.value)
            if consume('FLOAT'):
                return IR.DoubleLiteral(token, token.value)
            if at('NAME'):
                id = parse_id()
                with push(token):
                    defn = scope[id]

                if not isinstance(defn, IR.ScopeVariableValue):
                    with push(token), push(defn.token):
                        raise error(f'{id} is not a variable')

                return IR.Name(token, defn)
            if consume('@'):
                if at('STRING'):
                    value = expect('STRING').value
                    return IR.CStringLiteral(token, value)
                else:
                    expect('STRING')

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

            return IR.Include(token, use_quotes, name)

        def parse_import(includes):
            token = peek()
            use_from = bool(consume('from'))
            if not use_from:
                expect('import')

            module_parts = [parse_id()]
            while consume('.'):
                module_parts.append(parse_id())
            module_name = '.'.join(module_parts)
            incname = header_name_from_module_name(module_name)

            if not any(inc.value == incname for inc in includes):
                includes.append(IR.Include(token, True, incname))

            if module_name not in scope.module_scope_cache:
                assert module_name not in scope.translation_unit_cache, (
                    module_name
                )
                relpath = os.path.join(*module_parts) + '.k'
                path = os.path.join(scope.source_root, relpath)
                new_source = Source.from_path(path)
                module_scope = Scope(scope.parent)
                module_scope.module_name = module_name
                with push(token):
                    tu = parser.parse(new_source, module_scope)
                scope.translation_unit_cache[module_name] = tu
                scope.module_scope_cache[module_name] = module_scope
            else:
                module_scope = scope.module_scope_cache[module_name]

            if use_from:
                expect('import')
                exported_name = parse_id()
                with push(token):
                    exported_def = module_scope[exported_name]
            else:
                # To support importing module names, we need some kind
                # of ModuleType etc.
                assert False, 'TODO'

            if consume('as'):
                alias = parse_id()
            elif use_from:
                alias = exported_name
            else:
                alias = module_parts[-1]

            scope[alias] = exported_def

        def parse_translation_unit():
            token = peek()
            includes = []
            defs = []
            while True:
                consume_all('\n')
                if at('#'):
                    includes.append(parse_include())
                elif at('import') or at('from'):
                    parse_import(includes)
                else:
                    break
            consume_all('\n')
            while not at('EOF'):
                parse_global(defs)
                consume_all('\n')
            return IR.TranslationUnit(
                token, scope.module_name, includes, defs)

        return parse_translation_unit()

    def is_valid_module_name(module_name):
        return all(c.isalnum() or c in '._' for c in module_name) and (
            module_name.lower() == module_name
        )

    @ns
    def header_name_from_module_name(module_name):
        assert is_valid_module_name(module_name), module_name
        return f'{module_name}.k.h'

    @ns
    def source_name_from_module_name(module_name):
        assert is_valid_module_name(module_name), module_name
        return f'{module_name}.k.c'


@Namespace
def analyzer(ns):

    returns = Multimethod('returns')
    ns(returns, 'returns')

    @returns.on(IR.Block)
    def returns(self):
        return any(map(returns, self.statements))

    @returns.on(IR.LocalVariableDefinition)
    def returns(self):
        return False

    @returns.on(IR.Return)
    def returns(self):
        return True

    @returns.on(IR.ExpressionStatement)
    def returns(self):
        return False


@Namespace
def C(ns):
    class FractalStringBuilder(object):
        """String builder with additional features
        useful for generating C code.
        """
        def __init__(self, depth=0, parent=None):
            self.parts = []
            self.depth = depth
            self.parent = parent
            self.root = self if parent is None else parent.root
            self.decls = None if parent is None else parent.decls

            if self.root is self:
                self.next_tempvar_id = 0

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
            child = FractalStringBuilder(self.depth + depth_diff, self)
            self.parts.append(child)
            return child

        def new_tempvar_name(self):
            name = f'{UNIVERSAL_PREFIX}T{self.root.next_tempvar_id}'
            self.root.next_tempvar_id += 1
            return name

        def declare_var(self, type, name=None):
            name = self.new_tempvar_name() if name is None else name
            decl = declare(type, name)

            if type == IR.VarType():
                self.decls += f'{decl} = KLCnull;'
            elif isinstance(type, IR.PointerType):
                self.decls += f'{decl} = NULL;'
            elif type in IR.INTEGRAL_TYPES:
                self.decls += f'{decl} = 0;'
            else:
                # TODO: Zero initialize structs as well.
                self.decls += f'{decl};'

            return name

    encode_map = {
        '_': '_U',  # U for Underscore
        '.': '_D',  # D for Dot
        '#': '_H',  # H for Hashtag
    }

    def encode_char(c):
        if c.isalnum() or c.isdigit():
            return c
        elif c in encode_map:
            return encode_map[c]
        raise TypeError(f'Invalid name character {c}')

    @ns
    def encode(name, prefix=ENCODED_NAME_PREFIX):
        return prefix + ''.join(map(encode_char, name))

    @ns
    def write_out(tu: IR.TranslationUnit, outdir):
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

        header_name = parser.header_name_from_module_name(tu.name)
        header_content = header_for(tu)
        with open(os.path.join(outdir, header_name), 'w') as f:
            f.write(header_content)

        source_name = parser.source_name_from_module_name(tu.name)
        source_content = source_for(tu)
        with open(os.path.join(outdir, source_name), 'w') as f:
            f.write(source_content)

    @ns
    def header_for(tu: IR.TranslationUnit):
        out = FractalStringBuilder()
        header_name = parser.header_name_from_module_name(tu.name)
        header_macro = header_name.replace('.', '_').replace('/', '_')

        out += f'#ifndef {header_macro}'
        out += f'#define {header_macro}'

        out += f'#include "kcrt.h"'
        for inc in tu.includes:
            if inc.use_quotes:
                out += f'#include "{inc.value}"'
            else:
                out += f'#include <{inc.value}>'

        for sd in tu.definitions:
            # Forward declare structs.
            # Even extern structs, it doesn't hurt to declare them.
            if isinstance(sd, IR.StructDefinition):
                name = encode(sd.name)
                out += f'typedef struct {name} {name};'

        for defn in tu.definitions:
            if isinstance(defn, IR.FunctionDefinition):
                out += f'{proto_for(defn)};'
            elif isinstance(defn, IR.ClassDefinition):
                out += f'KLCvar {encode(defn.name)}();'

        for sd in tu.definitions:
            # We actually define the structs here.
            # Any extern structs, we assume are already defined
            # elsewhere.
            if isinstance(sd, IR.StructDefinition) and not sd.extern:
                name = encode(sd.name)
                out += f'struct {name}' '{'
                outf = out.spawn(1)
                out += '};'

                for field in sd.fields:
                    outf += declare(field.type, encode(field.name)) + ';'

        out += f'#endif/*{header_macro}*/'

        return str(out)

    @ns
    def source_for(tu: IR.TranslationUnit):
        out = FractalStringBuilder()
        header_name = parser.header_name_from_module_name(tu.name)

        out += f'#include "{header_name}"'

        for defn in tu.definitions:
            translate(defn, out)

        return str(out)

    proto_for = Multimethod('proto_for')

    @proto_for.on(IR.FunctionDefinition)
    def proto_for(fd):
        return declare(
            fd.type,
            fd.name if fd.extern else encode(fd.name),
            [p.name for p in fd.parameters])

    declare = Multimethod('declare')

    @declare.on(IR.FunctionType)
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
            declare(pt, encode(argname))
            for argname, pt in zip(argnames, ft.parameters)
        ] + (
            ['...'] if ft.vararg else []
        ))
        return declare(ft.return_type, name + f'({params})')

    @declare.on(IR.ConstType)
    def declare(ct, name: str):
        return declare(ct.base, 'const ' + name)

    @declare.on(IR.PointerType)
    def declare(pt, name: str):
        return declare(pt.base, '*' + name)

    @declare.on(IR.PrimitiveType)
    def declare(pt, name: str):
        return f'{pt.name} {name}'

    @declare.on(IR.VarType)
    def declare(vt, name: str):
        return f'KLCvar {name}'

    @declare.on(IR.StructType)
    def declare(st, name: str):
        return f'{encode(st.name)} {name}'

    translate = Multimethod('translate')

    @translate.on(IR.PointerType)
    def translate(pt):
        return f'{translate(pt.base)}*'

    @translate.on(IR.PrimitiveType)
    def translate(pt):
        return pt.name

    @translate.on(IR.StructDefinition)
    def translate(sd, out):
        # All the generation for structs are done in the header
        pass

    @translate.on(IR.ClassDefinition)
    def translate(cd, out):
        struct_name = encode(cd.name, prefix='KLCC')
        var_name = encode(cd.name, prefix='KLCV')

        out += f'struct {struct_name} ' '{'
        inner = out.spawn(1)
        out += '};'
        inner += f'KLCheader header;'

        encoded_name = encode(cd.name)

        out += f'static KLCXClass* {var_name} = NULL;'

        out += f'KLCvar {encode(cd.name)}()'
        out += '{'
        out += f'  if (!{var_name})' '{'
        inner = out.spawn(2)
        inner += f'{var_name} = KLCXNewClass("{cd.name}", NULL, NULL);'
        out += '  }'
        out += f'  return KLCXObjectToVar((KLCheader*) {var_name});'
        out += '}'

    @translate.on(IR.Parameter)
    def translate(param):
        return declare(param.type, encode(param.name))

    @translate.on(IR.FunctionDefinition)
    def translate(self, out):
        out += proto_for(self)
        declare_release_pool = True

        translate(
            self.body,
            out,
            declare_release_pool,
        )

    @translate.on(IR.Block)
    def translate(self, out, declare_release_pool=False):
        tempvar = out.new_tempvar_name()

        out += '{'
        inner = out.spawn(1)
        out += '}'

        if declare_release_pool:
            inner += 'KLCXReleasePool KLCXrelease_pool = {0, 0, NULL};'

        inner += f'size_t {tempvar} = KLCXrelease_pool.size;'

        # TODO: This is HACK. Factor this properly.
        inner.decls = inner.spawn()

        for stmt in self.statements:
            translate(stmt, inner)

        inner += f'KLCXResize(&KLCXrelease_pool, {tempvar});'

    @translate.on(IR.LocalVariableDefinition)
    def translate(self, out):
        cname = out.declare_var(self.type, encode(self.name))

        if self.expression:
            out += f'{cname} = {translate(self.expression)};'

    @translate.on(IR.Return)
    def translate(self, out):
        if self.expression is None:
            out += 'KLCXDrainPool(&KLCXrelease_pool);'
            out += 'return;'
        else:
            cname = out.declare_var(self.expression.expression_type)
            out += f'{cname} = {translate(self.expression)};'
            out += f'KLCXDrainPool(&KLCXrelease_pool);'
            out += f'return {cname};'

    @translate.on(IR.ExpressionStatement)
    def translate(self, out):
        out += translate(self.expression) + ';'

    @translate.on(IR.FunctionCall)
    def translate(self):
        f = translate(self.f)
        if not isinstance(self.f, IR.Name):
            f = f'({f})'
        args = ', '.join(map(translate, self.args))
        ret = f'{f}({args})'
        if self.expression_type == IR.VarType():
            ret = f'KLCXPush(&KLCXrelease_pool, {ret})'
        return ret

    @translate.on(IR.Name)
    def translate(self):
        return recall_name(self.definition)

    @translate.on(IR.IntLiteral)
    def translate(self):
        return str(self.value)

    @translate.on(IR.DoubleLiteral)
    def translate(self):
        return str(self.value)

    @translate.on(IR.CStringLiteral)
    def translate(self):
        s = (self.value
            .replace('\\', '\\\\')
            .replace('\t', '\\t')
            .replace('\n', '\\n')
            .replace('\r', '\\r')
            .replace('"', '\\"')
            .replace("'", "\\'"))
        return f'"{s}"'

    @translate.on(IR.Operation)
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

    # Given the declaration of a name, returns what that should
    # look like when used in translated C.
    recall_name = Multimethod('recall_name')

    @recall_name.on(IR.FunctionStub)
    def recall_name(self):
        return self.name if self.extern else encode(self.name)

    @recall_name.on(IR.ClassStub)
    def recall_name(self):
        return encode(self.name) + '()'

    @recall_name.on(IR.LocalVariableDefinition)
    def recall_name(self):
        return encode(self.name)

    @recall_name.on(IR.Parameter)
    def recall_name(self):
        return encode(self.name)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('path')
    argparser.add_argument('--out-dir', default='out')
    argparser.add_argument('--src-root', default='srcs')
    args = argparser.parse_args()
    global_scope = parser.new_global_scope()
    global_scope.source_root = os.path.abspath(args.src_root)
    source = Source.from_path(args.path)
    with global_scope.namespace(MAIN_MODULE_NAME) as scope:
        main_translation_unit = parser.parse(source=source, scope=scope)
    if os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    C.write_out(main_translation_unit, outdir=args.out_dir)
    for tu in scope.translation_unit_cache.values():
        C.write_out(tu, outdir=args.out_dir)

if __name__ == '__main__':
    main()
