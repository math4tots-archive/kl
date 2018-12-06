import argparse
import collections
import contextlib
import itertools
import os
import re
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


class FractalStringBuilder(object):
    def __init__(self, depth):
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


class Source(typing.NamedTuple):
    name: str
    filename: str
    data: str

    module_name_pattern = re.compile(r'^[a-z_]+(?:\.[a-z_]+)*$')

    @classmethod
    def from_name_and_path(cls, name, path):
        with open(path) as f:
            data = f.read()
        return cls(name, path, data)


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

    def map(self, mapping_function):
        return Promise(lambda: mapping_function(self.resolve()))

    def __repr__(self):
        return f'Promise({repr(self.resolve())})'

    @classmethod
    def value(cls, value):
        @Promise
        def promise():
            return value
        return promise


def pcall(f, *mixed_args):
    """Utility for calling functions when some arguments are still
    Promises.
    Semantically, this is kind of dirty mixing Promises and
    non-Promises like this, but in practice, at least for use
    cases here, I never actually want Promise types in the IR
    so this is ok.
    """
    @Promise
    def promise():
        return f(*[
            arg.resolve() if isinstance(arg, Promise) else arg
            for arg in mixed_args
        ])
    return promise


@Namespace
def typeutil(ns):
    @ns
    class FakeType(object):
        @property
        def __name__(self):
            return repr(self)

    class SingleArgumentFakeType(FakeType):
        def __init__(self, subtype):
            self.subtype = subtype

        def __getitem__(self, subtype):
            return type(self)(subtype)

        def __repr__(self):
            return f'{type(self).__name__}[{self.subtype.__name__}]'

    class ListType(SingleArgumentFakeType):
        def __instancecheck__(self, obj):
            return (
                isinstance(obj, list) and
                all(isinstance(x, self.subtype) for x in obj))

    List = ListType(object)
    ns(List, 'List')

    class NonEmptyListType(ListType):
        def __instancecheck__(self, obj):
            return super().__instancecheck__(obj) and obj

    NonEmptyList = NonEmptyListType(object)
    ns(NonEmptyList, 'NonEmptyList')

    class OptionalType(SingleArgumentFakeType):
        def __instancecheck__(self, obj):
            return obj is None or isinstance(obj, self.subtype)

    Optional = OptionalType(object)
    ns(Optional, 'Optional')

    class AndType(FakeType):
        def __init__(self, subtypes):
            self.subtypes = subtypes

        def __getitem__(self, subtypes):
            assert isinstance(subtypes, tuple), subtypes
            return type(self)(subtypes)

        def __repr__(self):
            subtypes = ','.join(map(repr, self.subtypes))
            return f'And[{subtypes}]'

        def __instancecheck__(self, obj):
            return all(isinstance(obj, t) for t in self.subtypes)

    And = AndType([object])
    ns(And, 'And')

    class OrType(FakeType):
        def __init__(self, subtypes):
            assert all(isinstance(t, type) for t in subtypes)
            self.subtypes = subtypes

        def __getitem__(self, subtypes):
            assert isinstance(subtypes, tuple), subtypes
            return type(self)(subtypes)

        def __repr__(self):
            subtypes = ','.join(map(repr, self.subtypes))
            return f'Or[{subtypes}]'

        def __instancecheck__(self, obj):
            return any(isinstance(obj, t) for t in self.subtypes)

    Or = OrType([object])
    ns(Or, 'Or')

    class NotType(SingleArgumentFakeType):
        def __instancecheck__(self, obj):
            return not isinstance(obj, self.subtype)

    Not = NotType(object)
    ns(Not, 'Not')


@Namespace
def lexer(ns):
    C_KEYWORDS = [
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
      'inline',  # (since C99)
      'int',
      'long',
      'register',
      'restrict',  # (since C99)
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
      '_Alignas',  # (since C11)
      '_Alignof',  # (since C11)
      '_Atomic',  # (since C11)
      '_Bool',  # (since C99)
      '_Complex',  # (since C99)
      '_Generic',  # (since C11)
      '_Imaginary',  # (since C99)
      '_Noreturn',  # (since C11)
      '_Static_assert',  # (since C11)
      '_Thread_local',
    ]

    C_PRIMITIVE_TYPE_SPECIFIERS = {
        'void',
        'char',
        'short',
        'int',
        'long',
        'float',
        'double',
        'signed',
        'unsigned',
    }
    ns(C_PRIMITIVE_TYPE_SPECIFIERS, 'C_PRIMITIVE_TYPE_SPECIFIERS')

    PRIMITIVE_TYPE_NAMES = (
        'void',
        'char',
        'short',
        'int',
        'long',
        'float',
        'double',
        'unsigned',

        # char is special in that 'char', 'unsigned char', and
        # 'signed char' are all distinct types.
        'unsigned char',
        'signed char',

        # all other 'signed' variants dupe other types.

        # 'unsigned int',  # dupes 'unsigned'
        'unsigned short',
        'unsigned long',
        # 'unsigned long long',  # not available in C89

        # 'long int',  # dupes 'long'
        'long double',
        # 'long long',  # not available in C89
    )
    ns(PRIMITIVE_TYPE_NAMES, 'PRIMITIVE_TYPE_NAMES')

    KEYWORDS = {
        'bool',
        'is', 'not', 'null', 'true', 'false', 'delete',
        'and', 'or', 'in',
        'inline', 'extern', 'class', 'trait', 'final', 'def', 'auto',
        'struct', 'const', 'throw',
        'for', 'if', 'else', 'while', 'break', 'continue', 'return',
        'with', 'from', 'import', 'as', 'try', 'catch', 'finally', 'raise',
        'except', 'case','switch', 'var', 'this',
        'static_cast',
    } | set(C_KEYWORDS)
    ns(KEYWORDS, 'KEYWORDS')

    SYMBOLS = tuple(reversed(sorted([
      '\n',
      '||', '&&', '|', '&', '<<', '>>', '~', '...', '$', '->',
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
        return lex(Source('main', None, data))

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
            if s.startswith(('r"', "r'", 'cr"', "cr'"), i):
                is_c_str = (s[i] == 'c')
                if is_c_str:
                    i += 1
                i += 1  # 'r'
                q = s[i:i+3] if s.startswith(s[i] * 3, i) else s[i]
                i += len(q)
                cut_start = i
                while i < len(s) and not s.startswith(q, i):
                    i += 1
                if i >= len(s):
                    raise error([a], 'Unterminated raw string literal')
                i += len(q)
                token_type = 'C_STRING' if is_c_str else 'STRING'
                yield mt(a, 'STRING', s[cut_start:i-len(q)])
                continue

            # normal string literal
            if s.startswith(('"', "'", 'c"', "c'"), i):
                is_c_str = (s[i] == 'c')
                if is_c_str:
                    i += 1
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
                token_type = 'C_STRING' if is_c_str else 'STRING'
                yield mt(a, token_type, ''.join(sb))
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


builtin_token = lexer.lex(Source(*(['builtin'] * 3)))[0]


@Namespace
def IR(ns):

    class LazyProperty:
        _type = None

        def __init__(self, name, type_callback, callback):
            self.name = name
            self.type_callback = type_callback
            self.callback = callback

        @property
        def type(self):
            if self._type is None:
                self._type = self.type_callback()
            return self._type

        def __get__(self, obj, type=None):
            if self.name not in obj.__dict__:
                obj.__dict__[self.name] = result = self.callback(obj)
                if not isinstance(result, self.type):
                    raise TypeError(
                        f'Expected property {self.name} to be '
                        f'{self.type} but got {result}')
            return obj.__dict__[self.name]

        def __set__(self, obj, value):
            raise AttributeError(self.name)

    def lazy(type_callback):
        def wrapper(f):
            return LazyProperty(f.__name__, type_callback, f)
        return wrapper

    @ns
    class Node(object):
        def __init__(self, token, *args, **kwargs):
            cls = type(self)
            self.token = token
            node_fields = list(cls.node_fields)

            def set_arg(fname, ftype, arg):
                if not isinstance(arg, ftype):
                    raise TypeError(
                        f'Expected type of {fname} for {cls.__name__} '
                        f'to be {ftype} but got {arg}')
                setattr(self, fname, arg)

            for key, arg in kwargs.items():
                for field in node_fields:
                    if field[0] == key:
                        break
                else:
                    raise TypeError(
                        f'{key} is not a field for {cls.__name__}')
                fname, ftype = field
                node_fields = [f for f in node_fields if f is not field]
                set_arg(fname, ftype, arg)

            if len(args) < len(node_fields):
                raise TypeError(f'Too few arguments for {cls.__name__}')
            elif len(args) > len(node_fields):
                raise TypeError(f'Too many arguments for {cls.__name__}')

            for (fname, ftype), arg in zip(node_fields, args):
                set_arg(fname, ftype, arg)

        def dump(self):
            return '%s(%s)' % (
                type(self).__name__,
                ', '.join(repr(getattr(self, n))
                for n, _ in type(self).node_fields),
            )

        def format(self, depth=0, out=None):
            return_ = out is None
            out = [] if out is None else out
            ind = '  ' * depth
            out.append(f'{ind}{type(self).__name__}\n')
            for fname, _ in type(self).node_fields:
                fval = getattr(self, fname)
                if isinstance(fval, list) and fval:
                    out.append(f'{ind}  {fname}...\n')
                    for item in fval:
                        if isinstance(item, Node):
                            item.format(depth + 2, out)
                        else:
                            out.append(f'{ind}    {repr(item)}\n')
                elif isinstance(fval, CollectionNode):
                    out.append(f'{ind}  {fname}\n')
                    fval.format(depth + 2, out)
                else:
                    out.append(f'{ind}  {fname} = {repr(fval)}\n')

            if return_:
                return ''.join(out)

        def foreach(self, f):
            self.map(f)

        def map(self, f):
            nt = type(self)
            return nt(self.token, *[
                Node._map_helper(f, getattr(self, fieldname))
                for fieldname, _ in nt.node_fields
            ])

        @classmethod
        def _map_helper(cls, f, value):
            if isinstance(value, (type(None), int, float, bool, str, IR.Type)):
                return value
            if isinstance(value, list):
                return [Node._map_helper(f, x) for x in value]
            if isinstance(value, tuple):
                return tuple(Node._map_helper(f, x) for x in value)
            if isinstance(value, set):
                return {Node._map_helper(f, x) for x in value}
            if isinstance(value, dict):
                return {
                    Node._map_helper(k): Node._map_helper(v)
                    for k, v in value.items()
                }
            if isinstance(value, Node):
                return f(value)
            raise TypeError(
                f'Unsupported Node.map element type '
                f'{type(value)} ({repr(value)})')

    @ns
    class CollectionNode(Node):
        """A marker class to help with Node.format"""

    @ns
    class GlobalDefinition(Node):
        pass

    @ns
    class Expression(Node):
        """
        abstract
            type: Type
        """

        @property
        def is_pseudo_expression(self):
            return isinstance(self, PseudoExpression)

        class WithType(typeutil.FakeType):

            def __init__(self, *types):
                self.types = types

            def __instancecheck__(self, obj):
                return (
                    isinstance(obj, Expression) and
                    obj.type in self.types
                )

            def __eq__(self, other):
                return (
                    type(self) is type(other) and
                    self.types == other.types
                )

            def __hash__(self):
                return hash((type(self), self.types))

            def __repr__(self):
                return f'WithType({self.types})'

        class WithTypeOfClass(typeutil.FakeType):

            def __init__(self, type_class):
                self.type_class = type_class

            def __instancecheck__(self, obj):
                return (
                    isinstance(obj, Expression) and
                    isinstance(obj.type, self.type_class)
                )

            def __eq__(self, other):
                return type(self) is type(other) and self.type == other.type

            def __hash__(self):
                return hash((type(self), self.type))

            def __repr__(self):
                return f'WithType({self.type})'

        class _WithVarType(typeutil.FakeType):
            def __instancecheck__(self, obj):
                return (
                    isinstance(obj, Expression) and
                    obj.type == VAR_TYPE
                )

        WITH_VAR_TYPE = _WithVarType()

    @ns
    class Type:
        """Marker class for what values indicate Types.
        """
        # Only 'declarable' types can be directly declared
        # (e.g. as local variable or field).
        # Most types are declarable, but function types are not,
        # and struct types are not until they are defined
        declarable = True

        @property
        def is_meta_type(self):
            return isinstance(self, PseudoType)

    @ns
    class Retainable:
        "Type mixin that indicates the given type is reference counted"

    class PseudoType(Type):
        """Type of expressions that aren't really expressions"""

    PSEUDO_TYPE = PseudoType()

    @ns
    class Declaration(Node):
        """Marker class for values that can be looked up by a Scope.

        abstract
            token: Token
        """

    @ns
    class TypeDeclaration(Type, Declaration):
        pass

    @ns
    class PseudoDeclaration(Declaration):
        """This is for injecting extra information into the scope
        PseudoDeclaration names must always start with '@'
        """

    @ns
    class VariableDeclaration(Declaration):
        pass

    @ns
    class BaseLocalVariableDeclaration(VariableDeclaration):
        node_fields = (
            ('type', Type),
            ('name', str),
        )

        mutable = False

    @ns
    class Parameter(BaseLocalVariableDeclaration):
        mutable = True

    @ns
    class ParameterList(Node):
        node_fields = (
            ('params', typeutil.List[Parameter]),
            ('vararg', bool),
        )

    @ns
    class FunctionDefinition(GlobalDefinition, Declaration):
        node_fields = (
            ('extern', bool),
            ('rtype', Type),
            ('module_name', str),
            ('short_name', str),
            ('plist', ParameterList),
            ('has_body', bool),
            ('body_promise', typeutil.Optional[Promise]),
        )

        @lazy(lambda: typeutil.Optional[Block])
        def body(self):
            return (
                self.body_promise.resolve()
                    if self.body_promise else None
            )

        @lazy(lambda: FunctionType)
        def type(self):
            return FunctionType(
                self.extern,
                self.rtype,
                [p.type for p in self.plist.params],
                self.plist.vararg,
            )

        def __repr__(self):
            return (
                f'FunctionDefinition({self.rtype}, '
                f'{self.module_name}#{self.short_name}, {self.plist})'
            )

    @ns
    class StructOrClassDefinition(GlobalDefinition, TypeDeclaration):
        _fields = None
        node_fields = (
            ('extern', bool),
            ('module_name', str),
            ('short_name', str),
            ('field_promises', typeutil.List[Promise]),
        )

        @lazy(lambda: typeutil.List[FieldDefinition])
        def fields(self):
            return [p.resolve() for p in self.field_promises]

    @ns
    class TraitOrClassDefinition:
        defined = False

        @lazy(lambda: typeutil.List[StaticMethodDefinition])
        def static_methods(self):
            return [p.resolve() for p in self.static_method_promises]

        @lazy(lambda: typeutil.List[InstanceMethodDefinition])
        def instance_methods(self):
            return [p.resolve() for p in self.instance_method_promises]

        @lazy(lambda: typeutil.List[InstanceMethodDefinition])
        def instance_method_closure(self):
            methods = []
            seen = set()

            for method in self.instance_methods:
                if method.name not in seen:
                    methods.append(method)
                seen.add(method.name)

            for trait in self.traits:
                for method in trait.instance_method_closure:
                    if method.name not in seen:
                        methods.append(method)
                    seen.add(method.name)

            return methods

        @lazy(lambda: typeutil.List[TraitDefinition])
        def traits(self):
            return self.traits_promise.resolve()

    @ns
    class StructDefinition(StructOrClassDefinition):
        declarable = False

    @ns
    class ClassDefinition(
            StructOrClassDefinition, Retainable, TraitOrClassDefinition):
        node_fields = StructOrClassDefinition.node_fields + (
            ('delete_hook_promise', Promise),
            ('static_method_promises', typeutil.List[Promise]),
            ('instance_method_promises', typeutil.List[Promise]),
            ('traits_promise', Promise),
        )

        def __repr__(self):
            return f'ClassDefinition({self.module_name}#{self.short_name})'

        @lazy(lambda: DeleteHook)
        def delete_hook(self):
            return self.delete_hook_promise.resolve()

    @ns
    class TraitDefinition(
            GlobalDefinition, Declaration, TraitOrClassDefinition):
        extern = False
        declarable = False
        node_fields = (
            ('module_name', str),
            ('short_name', str),
            ('static_method_promises', typeutil.List[Promise]),
            ('instance_method_promises', typeutil.List[Promise]),
            ('traits_promise', Promise),
        )

    @ns
    class StaticMethodDefinition(Declaration):
        node_fields = (
            ('cls_promise', Promise),
            ('rtype_promise', Promise),
            ('name', str),
            ('plist_promise', Promise),
            ('body_promise', Promise),
        )

        @lazy(lambda: FunctionType)
        def type(self):
            return FunctionType(
                extern=False,
                rtype=self.rtype,
                paramtypes=[p.type for p in self.plist.params],
                vararg=self.plist.vararg,
            )

        @lazy(lambda: Type)
        def rtype(self):
            return self.rtype_promise.resolve()

        @lazy(lambda: TraitOrClassDefinition)
        def cls(self):
            return self.cls_promise.resolve()

        @lazy(lambda: ParameterList)
        def plist(self):
            return self.plist_promise.resolve()

        @lazy(lambda: Block)
        def body(self):
            return self.body_promise.resolve()

    @ns
    class InstanceMethodDefinition(Declaration):
        node_fields = (
            ('cls_promise', Promise),
            ('rtype_promise', Promise),
            ('name', str),
            ('plist_promise', Promise),  # does not include 'this'
            ('body_promise', Promise),
        )

        @lazy(lambda: Type)
        def rtype(self):
            return self.rtype_promise.resolve()

        @lazy(lambda: TraitOrClassDefinition)
        def cls(self):
            return self.cls_promise.resolve()

        @lazy(lambda: typeutil.Or[VarType, ClassDefinition])
        def this_type(self):
            return (
                VAR_TYPE
                if isinstance(self.cls, TraitDefinition) else
                self.cls
            )

        @lazy(lambda: ParameterList)
        def plist(self):
            return self.plist_promise.resolve()

        @lazy(lambda: Block)
        def body(self):
            return self.body_promise.resolve()

    @ns
    class DeleteHook(Node):
        node_fields = (
            ('body_promise', Promise),
        )

        @lazy(lambda: Block)
        def body(self):
            return self.body_promise.resolve()

    @ns
    class GlobalVariableDefinition(GlobalDefinition, VariableDeclaration):
        node_fields = (
            ('extern', bool),
            ('type', Type),
            ('module_name', str),
            ('short_name', str),
            ('expr', Expression),
        )

        def __repr__(self):
            return (
                f'GlobalVariableDefinition('
                f'{self.module_name}#{self.short_name})'
            )

    @ns
    class LocalVariableDeclaration(BaseLocalVariableDeclaration):
        mutable = True

    @ns
    class DeleteQueueDeclaration(VariableDeclaration):
        node_fields = ()

    @ns
    class PrimitiveTypeDefinition(TypeDeclaration, GlobalDefinition):
        node_fields = (
            ('name', str),
        )

        def __repr__(self):
            return f'PrimitiveTypeDefinition({self.name})'

        def __eq__(self, other):
            return type(self) is type(other) and self.name == other.name

        def __hash__(self):
            return hash((type(self), self.name))

    class ProxyMixin:
        def __eq__(self, other):
            return type(self) is type(other) and self._proxy == other._proxy

        def __hash__(self):
            return hash((type(self), self._proxy))

        def __repr__(self):
            return f'{type(self).__name__}{self._proxy}'

    @ns
    class PointerType(Type, ProxyMixin):
        def __init__(self, base):
            self.base = base

        @property
        def _proxy(self):
            return (self.base,)

    @ns
    class FunctionType(Type, ProxyMixin):
        declarable = False

        def __init__(self, extern, rtype, paramtypes, vararg):
            self.extern = extern
            self.rtype = rtype
            self.paramtypes = tuple(paramtypes)
            self.vararg = vararg
            assert isinstance(
                list(self.paramtypes),
                typeutil.List[Type],
            ), self.paramtypes

        @property
        def _proxy(self):
            return (self.extern, self.rtype, self.paramtypes, self.vararg)

    @ns
    def convert_args(self, scope, token, args):
        if self.vararg:
            if len(self.paramtypes) > len(args):
                with scope.push(token):
                    raise scope.error(
                        f'Expected at least {len(self.paramtypes)} '
                        f'args but got {len(args)}'
                    )
        else:
            if len(self.paramtypes) != len(args):
                with scope.push(token):
                    raise scope.error(
                        f'Expected {len(self.paramtypes)} '
                        f'args but got {len(args)}'
                    )
        converted_args = []
        for i, (pt, arg) in enumerate(zip(self.paramtypes, args), 1):
            converted_args.append(scope.convert(arg, pt))
        converted_args.extend(args[len(self.paramtypes):])
        for arg in converted_args:
            arg.type  # validate that this field is set for all args
        return converted_args

    @ns
    class ConstType(Type, ProxyMixin):
        def __init__(self, base):
            self.base = base

        @property
        def _proxy(self):
            return (self.base,)

    STRING_CONVERSION_FUNCTION_NAME = '%str'
    ns(STRING_CONVERSION_FUNCTION_NAME, 'STRING_CONVERSION_FUNCTION_NAME')

    NEW_LIST_FUNCTION_NAME = '%mklist'
    ns(NEW_LIST_FUNCTION_NAME, 'NEW_LIST_FUNCTION_NAME')

    LIST_PUSH_FUNCTION_NAME = '%listpush'
    ns(LIST_PUSH_FUNCTION_NAME, 'LIST_PUSH_FUNCTION_NAME')

    def nptype(c_name):
        return PrimitiveTypeDefinition(builtin_token, c_name)

    VOID = nptype('void')
    ns(VOID, 'VOID')

    BOOL = nptype('KLC_bool')
    ns(BOOL, 'BOOL')

    INT = nptype('KLC_int')
    ns(INT, 'INT')

    FLOAT = nptype('KLC_float')
    ns(FLOAT, 'FLOAT')

    VOIDP = PointerType(VOID)
    ns(VOIDP, 'VOIDP')

    c_type_name_map = {
        'c_char': 'char',
        'c_unsigned_char': 'unsigned char',
        'c_signed_char': 'signed char',
        'c_short': 'short',
        'c_unsigned_short': 'unsigned short',
        'c_int': 'int',
        'c_unsigned_int': 'unsigned int',
        'c_long': 'long',
        'c_unsigned_long': 'unsigned long',
        'size_t': 'size_t',
        'ptrdiff_t': 'ptrdiff_t',
        'c_float': 'float',
        'c_double': 'double',
        'type': 'KLC_Class*',
    }

    c_type_map = {
        kc_name: nptype(c_name)
            for kc_name, c_name in c_type_name_map.items()
    }
    ns(c_type_map, 'c_type_map')

    for kc_name, ptype in c_type_map.items():
        ns(ptype, kc_name)

    TYPE = c_type_map['type']
    ns(TYPE, 'TYPE')

    CHAR_TYPES = {
        c_type_map['c_char'],
        c_type_map['c_signed_char'],
        c_type_map['c_unsigned_char'],
    }
    ns(CHAR_TYPES, 'CHAR_TYPES')

    INTEGRAL_TYPES = CHAR_TYPES | {
        INT,
        c_type_map['c_short'],
        c_type_map['c_unsigned_short'],
        c_type_map['c_int'],
        c_type_map['c_unsigned_int'],
        c_type_map['c_long'],
        c_type_map['c_unsigned_long'],
        c_type_map['size_t'],
        c_type_map['ptrdiff_t'],
    }
    ns(INTEGRAL_TYPES, 'INTEGRAL_TYPES')

    FLOAT_TYPES = {
        FLOAT,
        c_type_map['c_float'],
        c_type_map['c_double'],
    }
    ns(FLOAT_TYPES, 'FLOAT_TYPES')

    NUMERIC_TYPES = INTEGRAL_TYPES | FLOAT_TYPES
    ns(NUMERIC_TYPES, 'NUMERIC_TYPES')

    PRIMITIVE_TRUTHY_TYPES = {BOOL} | NUMERIC_TYPES
    ns(PRIMITIVE_TRUTHY_TYPES, 'PRIMITIVE_TRUTHY_TYPES')

    VAR_CONVERTIBLE_TYPES = {VOID, BOOL, TYPE} | NUMERIC_TYPES

    class VarType(Type, Retainable):
        pass

    VAR_TYPE = VarType()
    ns(VAR_TYPE, 'VAR_TYPE')

    @ns
    def to_value_expr(*, expr, scope):
        if isinstance(expr, ValueExpression):
            return expr

        if isinstance(expr, InstanceMethodReference):
            return IR.InstanceMethodCall(
                expr.token,
                f'__GET{expr.name}',
                [expr.owner],
            )

        with scope.push(expr.token):
            raise scope.error(f'{expr} is not a value expression')

    @ns
    def convert(*, expr, dest_type, scope):

        value_expr = to_value_expr(expr=expr, scope=scope)
        assert isinstance(value_expr, ValueExpression), value_expr

        if value_expr.type == dest_type:
            return value_expr

        if value_expr.type == VOID:
            with scope.push(expr.token):
                scope.error(f'Got void type but expected {dest_type}')

        return _convert(value_expr.type, dest_type, value_expr, scope)

    _convert = Multimethod('_convert', 3)
    ns(_convert, '_convert')

    PTD = PrimitiveTypeDefinition

    def convert_error(st, dt, expr, scope):
        with scope.push(expr.token):
            return scope.error(f'{st} is not convertible to {dt}')

    @_convert.on(PTD, PTD, Expression)
    def _convert(st, dt, expr, scope):
        if st == dt:
            return expr

        if dt == VOID:
            return Cast(expr.token, expr, dt)

        pair = (st.name, dt.name)

        if st in NUMERIC_TYPES and dt == BOOL:
            # This is a bit of a hack, considering that '!!'
            # is actually two operators.
            # TODO: Figure out a better story for tuthiness.
            return PrimitiveUnop(
                expr.token,
                BOOL,
                '!!',
                expr,
            )

        if st in INTEGRAL_TYPES and dt == INT:
            return Cast(expr.token, expr, dt)

        if st in FLOAT_TYPES and dt == FLOAT:
            return Cast(expr.token, expr, dt)

        if st in CHAR_TYPES and dt in CHAR_TYPES:
            return Cast(expr.token, expr, dt)

        raise convert_error(st, dt, expr, scope)

    @_convert.on(VarType, VarType, Expression)
    def _convert(st, dt, expr, scope):
        return expr

    @_convert.on(VarType, PTD, Expression)
    def _convert(st, dt, expr, scope):
        if dt in VAR_CONVERTIBLE_TYPES:
            return Cast(expr.token, expr, dt)
        raise convert_error(st, dt, expr, scope)

    @_convert.on(VarType, ClassDefinition, Expression)
    def _convert(st, dt, expr, scope):
        return Cast(expr.token, expr, dt)

    @_convert.on(ClassDefinition, VarType, Expression)
    def _convert(st, dt, expr, scope):
        return Cast(expr.token, expr, dt)

    @_convert.on(PTD, VarType, Expression)
    def _convert(st, dt, expr, scope):
        if st in VAR_CONVERTIBLE_TYPES:
            return Cast(expr.token, expr, dt)
        raise convert_error(st, dt, expr, scope)

    @_convert.on(PointerType, PointerType, Expression)
    def _convert(st, dt, expr, scope):
        if st == dt:
            return expr

        if dt == VOIDP:
            return Cast(expr.token, expr, dt)

        if isinstance(dt.base, ConstType) and st.base == dt.base.base:
            return Cast(expr.token, expr,  dt)

        raise convert_error(st, dt, expr, scope)

    @_convert.on(Type, Type, Expression)
    def _convert(st, dt, expr, scope):
        raise convert_error(st, dt, expr, scope)

    @ns
    class PseudoExpression(Expression):
        """Expressions that are not themselves concrete values.
        """

        @property
        def type(self):
            return PSEUDO_TYPE

    ValueExpression = typeutil.And[
        Expression,
        typeutil.Not[PseudoExpression],
    ]

    @ns
    class TraitOrClassName(Expression):
        type = TYPE

        node_fields = (
            ('cls', TraitOrClassDefinition),
        )

        def __repr__(self):
            return (
                f'TraitOrClassName('
                f'{self.cls.module_name}.{self.cls.short_name})'
            )

    @ns
    class StaticMethodName(PseudoExpression):
        node_fields = (
            ('defn', StaticMethodDefinition),
        )

    @ns
    class Block(Expression, CollectionNode):
        node_fields = (
            ('decls', typeutil.List[LocalVariableDeclaration]),
            ('exprs', typeutil.List[ValueExpression]),
        )

        @property
        def type(self):
            return self.exprs[-1].type if self.exprs else VOID

    @ns
    class Cast(Expression):
        node_fields = (
            ('expr', ValueExpression),
            ('type', Type),
        )

    @ns
    class FunctionName(PseudoExpression):
        node_fields = (
            ('defn', FunctionDefinition),
        )

        @property
        def type(self):
            return self.defn.type

        def __repr__(self):
            return (
                f'FunctionName({self.defn.module_name}.'
                f'{self.defn.short_name})'
            )

    @ns
    class DeleteQueueName(Expression):
        node_fields = ()
        type = VOIDP

    @ns
    class GlobalName(Expression):
        node_fields = (
            ('defn', GlobalVariableDefinition),
        )

        @property
        def type(self):
            return self.defn.type

    @ns
    class SetGlobalName(Expression):
        node_fields = (
            ('defn', GlobalVariableDefinition),
            ('expr', ValueExpression),
        )

        @property
        def type(self):
            return self.defn.type

    @ns
    class LocalName(Expression):
        node_fields = (
            ('decl', BaseLocalVariableDeclaration),
        )

        @property
        def type(self):
            return self.decl.type

    @ns
    class SetLocalName(Expression):
        node_fields = (
            ('decl', BaseLocalVariableDeclaration),
            ('expr', ValueExpression),
        )

        @property
        def type(self):
            return self.decl.type

    @ns
    class FieldDefinition(Declaration):
        node_fields = (
            ('extern', bool),
            ('type', Type),
            ('name', str),
        )

    @ns
    class GetStructField(Expression):
        node_fields = (
            ('expr', ValueExpression),
            ('field_defn', FieldDefinition),
        )

        @property
        def type(self):
            return self.field_defn.type

    @ns
    class SetStructField(Expression):
        node_fields = (
            ('expr', Expression),
            ('field_defn', FieldDefinition),
            ('valexpr', Expression),
        )

        @property
        def type(self):
            return self.field_defn.type

    @ns
    class GetClassField(Expression):
        node_fields = (
            ('expr', Expression),
            ('field_defn', FieldDefinition),
        )

        @property
        def type(self):
            return self.field_defn.type

    @ns
    class SetClassField(Expression):
        node_fields = (
            ('field_defn', FieldDefinition),
            ('expr', Expression),
            ('valexpr', Expression),
        )

        @property
        def type(self):
            return self.field_defn.type

    @ns
    class FunctionCall(Expression):
        node_fields = (
            ('type', Type),
            ('f', Expression),
            ('args', typeutil.List[Expression]),
        )

    @ns
    class StaticMethodCall(Expression):
        node_fields = (
            ('f', StaticMethodDefinition),
            ('args', typeutil.List[Expression]),
        )

        @property
        def type(self):
            return self.f.rtype

    @ns
    class InstanceMethodReference(PseudoExpression):
        type = VAR_TYPE

        node_fields = (
            ('owner', Expression.WITH_VAR_TYPE),
            ('name', str),
        )

    @ns
    class InstanceMethodCall(Expression):
        type = VAR_TYPE

        node_fields = (
            ('name', str),
            ('args', typeutil.NonEmptyList[Expression.WITH_VAR_TYPE]),
        )

    @ns
    class PrimitiveUnop(Expression):
        node_fields = (
            ('type', PrimitiveTypeDefinition),
            ('op', str),
            ('expr', Expression),
        )

    @ns
    class PrimitiveBinop(Expression):
        node_fields = (
            ('type', PrimitiveTypeDefinition),
            ('op', str),
            ('left', Expression),
            ('right', Expression),
        )

    @ns
    class PointerAndIntArithmetic(Expression):
        node_fields = (
            ('type', PointerType),
            ('left', Expression.WithTypeOfClass(PointerType)),
            ('op', str),
            ('right', Expression.WithType(INT)),
        )

    @ns
    class If(Expression):
        node_fields = (
            ('cond', Expression.WithType(BOOL)),
            ('left', Expression),
            ('right', Expression),
        )

        @lazy(lambda: Type)
        def type(self):
            if self.left.type == self.right.type:
                return self.left.type
            else:
                return IR.VOID

    @ns
    class While(Expression):
        node_fields = (
            ('cond', Expression.WithType(BOOL)),
            ('body', Block),
        )

        @lazy(lambda: Type)
        def type(self):
            return VOID

    @ns
    class LogicalAnd(Expression):
        node_fields = (
            ('left', Expression.WithType(BOOL)),
            ('right', Expression.WithType(BOOL)),
        )

        type = BOOL

    @ns
    class LogicalOr(Expression):
        node_fields = (
            ('left', Expression.WithType(BOOL)),
            ('right', Expression.WithType(BOOL)),
        )

        type = BOOL

    @ns
    class Malloc(Expression):
        node_fields = (
            ('type', Type),
        )

    @ns
    class This(Expression):
        node_fields = (
            ('type', typeutil.Or[ClassDefinition, VarType]),
        )

        def __repr__(self):
            if isinstance(self.type, ClassDefinition):
                return f'This({self.type.module_name}.{self.type.short_name})'
            elif self.type == VAR_TYPE:
                return f'This(var)'
            assert False, self.type

    @ns
    class IntLiteral(Expression):
        type = INT

        node_fields = (
            ('value', int),
        )

    @ns
    class FloatLiteral(Expression):
        type = FLOAT

        node_fields = (
            ('value', float),
        )

    @ns
    class CStringLiteral(Expression):
        node_fields = (
            ('value', str),
        )

        @lazy(lambda: Type)
        def type(self):
            return PointerType(ConstType(IR.c_char))

    @ns
    class ThrowStringLiteral(Expression):
        type = VOID

        node_fields = (
            ('value', str),
        )

    @ns
    class Include(Node):
        node_fields = (
            ('use_quotes', bool),
            ('value', str),
        )

    @ns
    class FromImport(Node):
        node_fields = (
            ('module_name', str),
            ('exported_name', str),
            ('alias', str),
        )

    @ns
    class Module(Node):
        node_fields = (
            ('name', str),
            ('includes', typeutil.List[Include]),
            ('imports', typeutil.List[FromImport]),
            ('definitions', typeutil.List[GlobalDefinition]),
        )

    @ns
    class ExpressionContext(PseudoDeclaration):
        """Contains information needed for resolving expressions
        """
        node_fields = (
            # Indicates whether we're in an 'extern' context.
            # E.g. we cannot throw exceptions from an extern
            # context. Also the calling convention is different.
            ('extern', bool),

            ('this_type',
                typeutil.Optional[typeutil.Or[ClassDefinition, VarType]]),
        )

    def new_builtin_extern_struct(name):
        defn = StructDefinition(
            builtin_token,
            True,
            'builtin',
            name,
            [],
        )
        return defn

    HEADER_TYPE = new_builtin_extern_struct('KLC_Header')
    ns(HEADER_TYPE, 'HEADER_TYPE')
    ns(PointerType(HEADER_TYPE), 'HEADER_POINTER_TYPE')

    CLASS_TYPE = new_builtin_extern_struct('KLC_Class')
    ns(CLASS_TYPE, 'CLASS_TYPE')
    ns(PointerType(CLASS_TYPE), 'CLASS_POINTER_TYPE')

    ERROR_TYPE = new_builtin_extern_struct('KLC_Error')
    ns(ERROR_TYPE, 'ERROR_TYPE')
    ns(PointerType(ERROR_TYPE), 'ERROR_POINTER_TYPE')

    STACK_TYPE = new_builtin_extern_struct('KLC_Stack')
    ns(STACK_TYPE, 'STACK_TYPE')
    ns(PointerType(STACK_TYPE), 'STACK_POINTER_TYPE')


@Namespace
def parser(ns):

    class Scope:
        def __init__(self, parent, search_dir=None):
            if search_dir is None:
                search_dir = parent.search_dir
            self.parent = parent
            self.table = dict()
            self.search_dir = search_dir
            self.root = self if parent is None else parent.root
            self.stack = [] if parent is None else parent.stack
            self.cache = dict() if parent is None else parent.cache

        def error(self, message):
            raise Error(self.stack, message)

        @contextlib.contextmanager
        def push(self, token):
            self.stack.append(token)
            try:
                yield
            finally:
                self.stack.pop()

        def _check_key_value(self, key: str, value: IR.Declaration):
            assert isinstance(value, IR.Declaration), value
            assert (
                key.startswith('@') ==
                    isinstance(value, IR.PseudoDeclaration)
            ), (key, value)

        def __getitem__(self, key: str) -> IR.Declaration:
            value = self._getp(key).resolve()
            self._check_key_value(key, value)
            return value

        def _getp(self, key: str) -> Promise:
            if key in self.table:
                return self.table[key]
            elif self.parent is not None:
                return self.parent._getp(key)
            else:
                raise self.error(f'{repr(key)} not defined')

        def __setitem__(self, key: str, value: IR.Declaration):
            self._check_key_value(key, value)
            self.set_promise(value.token, key, Promise.value(value))

        def set_promise(self, token: Token, key: str, p: Promise):
            if key in self.table:
                with self.push(token), self.push(self.table[key].token):
                    raise self.error(f'Duplicate definition of {repr(key)}')
            self.table[key] = p

        def __contains__(self, key: str) -> bool:
            return key in self.table or self.parent and key in self.parent

        def check_module_name(self, module_name: str):
            if not Source.module_name_pattern.match(module_name):
                raise self.error(
                    f'Module names may only contain components with '
                    f'lower case letters, digits and underscores separated '
                    f'by dots ({repr(module_name)} is not a valid '
                    f'module name)')

        def _load(self, module_name) -> ('Scope', Promise):
            if module_name not in self.cache:
                self.check_module_name(module_name)
                path = os.path.join(
                    self.search_dir,
                    module_name.replace('.', os.path.sep) + '.k',
                )
                try:
                    with open(path) as f:
                        data = f.read()
                except IOError as e:
                    raise self.error(
                        f'Could not read {module_name} ({path})')
                self.cache[module_name] = _parse(
                    self.root,
                    Source(
                        name=module_name,
                        filename=path,
                        data=data,
                    ),
                )
            return self.cache[module_name]

        def load_scope_for(self, module_name):
            return self._load(module_name)[0]

        def load_promise_for(self, module_name):
            return self._load(module_name)[1]

        def convert(self, expr, type):
            """
            Returns a version of expr that's ensured to be of type 'type'.
            If this conversion is not allowed, raises an error.
            """
            return IR.convert(expr=expr, dest_type=type, scope=self)

        def from_import(self, token, module_name, exported_name, alias):
            with self.push(token):
                exported_defn = (
                    self.load_scope_for(module_name)[exported_name]
                )
            self[alias] = exported_defn
            return Promise.value(
                IR.FromImport(token, module_name, exported_name, alias),
            )

        def get_this_type(self):
            this_type = self['@ec'].this_type
            if this_type is None:
                raise scope.error(f'Not inside class instance context')
            return this_type

    BUILTINS_MODULE_NAME = 'builtins'
    ROOT_TRAIT_NAME = 'Object'

    _builtins_export_names = (
        IR.STRING_CONVERSION_FUNCTION_NAME,
        IR.NEW_LIST_FUNCTION_NAME,
        IR.LIST_PUSH_FUNCTION_NAME,
        'String',
        'StringBuilder',
        'List',
        'print',
        'str',
    )

    @ns
    def parse(
            source: Source,
            search_dir: str) -> typing.Dict[str, IR.Module]:
        global_scope = Scope(None, search_dir=search_dir)
        result = _parse(global_scope, source)
        global_scope.cache[source.name] = result
        return {
            name: p.resolve() for
                name, (scope, p) in
                global_scope.cache.items()
        }

    def _parse(global_scope: Scope, source: Source) -> (Scope, Promise):
        """
        Implementation notes:

            Functions with names matching 'expect_*'
                parses a construct, and returns some value.

            Functions with names matching 'parse_*'
                are counterparts to 'expect_*' functions, but
                return Promise types.

            Functions with names matching 'promise_*'
                are helper functions (usually for 'parse_*' functions)
                that don't do any parsing themselves, but
                return Promise types.
        """
        assert global_scope.parent is None
        module_scope = Scope(global_scope)
        module_name = source.name
        tokens = lexer.lex(source)
        i = 0
        indent_stack = []

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
                with module_scope.push(peek()):
                    raise module_scope.error(
                        f'Expected {repr(t)} but got {peek()}')
            return gettok()

        def expect_id():
            return expect('NAME').value

        def parse_import(scope):
            token = peek()
            expect('from')
            parts = [expect_id()]
            while consume('.'):
                parts.append(expect_id())
            module_name = '.'.join(parts)
            expect('import')
            exported_name = expect_id()
            alias = expect_id() if consume('as') else exported_name
            return scope.from_import(token, module_name, exported_name, alias)

        def expect_name(name):
            if peek().type != 'NAME' or peek().value != name:
                with module_scope.push(peek()):
                    raise module_scope.error(f'Expected name {name}')
            return gettok()

        def promise_type_from_name(scope, token, name):
            @Promise
            def promise():
                if name not in scope and name in IR.c_type_map:
                    return IR.c_type_map[name]
                with scope.push(token):
                    type = scope[name]
                if not isinstance(type, IR.Type):
                    with scope.push(token), scope.push(type):
                        raise scope.error(f'{name} is not a type')
                return type
            return promise

        def parse_type(scope):
            type_promise = parse_root_type(scope)
            while True:
                token = peek()
                if consume('*'):
                    type_promise = pcall(IR.PointerType, type_promise)
                elif consume('const'):
                    type_promise = pcall(IR.ConstType, type_promise)
                else:
                    break
            return type_promise

        def parse_root_type(scope):
            token = peek()
            if consume('bool'):
                return Promise.value(IR.BOOL)
            elif consume('int'):
                return Promise.value(IR.INT)
            elif consume('float'):
                return Promise.value(IR.FLOAT)
            elif consume('void'):
                return Promise.value(IR.VOID)
            elif consume('var'):
                return Promise.value(IR.VAR_TYPE)
            name = expect_id()
            return promise_type_from_name(scope, token, name)

        def parse_param_list(scope):
            token = expect('(')
            with skipping_newlines(True):
                paramps = []
                vararg = False
                while not consume(')'):
                    if consume('...'):
                        expect(')')
                        vararg = True
                        break
                    ptok = peek()
                    ptypep = parse_type(scope)
                    pname = expect_id()
                    param_promise = pcall(IR.Parameter, ptok, ptypep, pname)
                    scope.set_promise(ptok, pname, param_promise)
                    paramps.append(param_promise)
                    if not consume(','):
                        expect(')')
                        break
                return Promise(lambda: IR.ParameterList(
                token, [p.resolve() for p in paramps], vararg,
            ))

        def at_variable_declaration():
            if peek().type in ('bool', 'int', 'float', 'void', 'var'):
                return True

            # Whitelist a few patterns as being the start
            # of a variable declaration
            seqs = [
                ['NAME', 'NAME', '='],
                ['NAME', 'NAME', '\n'],
                ['NAME', '*', '*'],
                ['NAME', '*', 'NAME', '='],
                ['NAME', '*', 'NAME', '\n'],
                ['NAME', 'const'],
            ]
            for seq in seqs:
                if [t.type for t in tokens[i:i+len(seq)]] == seq:
                    return True

            return False

        def promise_truthy(scope, token, expr_promise):
            @Promise
            def promise():
                return scope.convert(expr_promise.resolve(), IR.BOOL)
            return promise

        def promise_set_local_name(scope, token, decl_promise, expr_promise):
            @Promise
            def promise():
                decl = decl_promise.resolve()
                expr = expr_promise.resolve()
                return process_assignment(
                    scope, token, IR.LocalName(token, decl), expr)
                return IR.SetLocalName(
                    token, decl, scope.convert(expr, decl.type))
            return promise

        def check_concrete_expression(scope, expr):
            if expr.type.is_meta_type or expr.is_pseudo_expression:
                with scope.push(expr.token):
                    raise scope.error(f'{expr} is not a value')
            return expr

        def promise_block(token, scope, decl_promises, expr_promises):
            @Promise
            def promise():
                _check_has_expression_context(scope)
                return IR.Block(
                    token,
                    [p.resolve() for p in decl_promises],
                    [
                        check_concrete_expression(scope, p.resolve())
                        for p in expr_promises
                    ],
                )
            return promise

        def parse_block(parent_scope):
            scope = Scope(parent_scope)
            token = expect('{')
            decl_promises = []
            expr_promises = []
            with skipping_newlines(False):
                consume_all('\n')
                while not consume('}'):
                    if at_variable_declaration():
                        decl_token = peek()
                        decl_type_promise = parse_type(scope)
                        decl_name = expect_id()
                        expr_promise = (
                            parse_expression(scope) if consume('=') else None
                        )
                        expect('\n')
                        consume_all('\n')
                        decl_promise = pcall(
                            IR.LocalVariableDeclaration,
                            decl_token,
                            decl_type_promise,
                            decl_name,
                        )
                        decl_promises.append(decl_promise)
                        scope.set_promise(
                            decl_token, decl_name, decl_promise)
                        if expr_promise is not None:
                            expr_promises.append(promise_set_local_name(
                                scope=scope,
                                token=decl_token,
                                decl_promise=decl_promise,
                                expr_promise=expr_promise,
                            ))
                    else:
                        expr_promises.append(parse_expression(scope))
                        expect('\n')
                    consume_all('\n')
            return promise_block(
                token=token,
                scope=scope,
                decl_promises=decl_promises,
                expr_promises=expr_promises,
            )

        def _check_has_expression_context(scope):
            assert isinstance(scope['@ec'], IR.ExpressionContext)

        def parse_truthy_expression(scope):
            token = peek()
            return promise_truthy(scope, token, parse_expression(scope))

        def parse_expression(scope):
            promise = parse_expression_or_pseudo_expression(scope)
            return promise.map(lambda expr: IR.to_value_expr(
                expr=expr,
                scope=scope,
            ))

        def parse_expression_or_pseudo_expression(scope):
            promise = parse_logical_or(scope)
            @promise.map
            def promise(expr):
                _check_has_expression_context(scope)
                return expr
            return promise

        def parse_logical_or(scope):
            expr_promise = parse_logical_and(scope)
            while True:
                token = peek()
                if consume('||'):
                    left_promise = promise_truthy(
                        scope,
                        token,
                        expr_promise,
                    )
                    right_promise = promise_truthy(
                        scope,
                        token,
                        parse_logical_and(scope),
                    )
                    expr_promise = promise_binop(
                        scope,
                        token,
                        '||',
                        left_promise,
                        right_promise,
                    )
                else:
                    break
            return expr_promise

        def parse_logical_and(scope):
            expr_promise = parse_comparison(scope)
            while True:
                token = peek()
                if consume('&&'):
                    left_promise = promise_truthy(
                        scope,
                        token,
                        expr_promise,
                    )
                    right_promise = promise_truthy(
                        scope,
                        token,
                        parse_comparison(scope),
                    )
                    expr_promise = promise_binop(
                        scope,
                        token,
                        '&&',
                        left_promise,
                        right_promise,
                    )
                else:
                    break
            return expr_promise

        def parse_comparison(scope):
            expr_promise = parse_additive(scope)
            while True:
                token = peek()
                for op in ('==', '!=', '<', '>', '<=', '>='):
                    if consume(op):
                        right_promise = parse_additive(scope)
                        expr_promise = promise_binop(
                            scope,
                            token,
                            op,
                            expr_promise,
                            right_promise,
                        )
                        break
                else:
                    break
            return expr_promise

        def parse_additive(scope):
            expr_promise = parse_multiplicative(scope)
            while True:
                token = peek()
                for op in ('+', '-'):
                    if consume(op):
                        right_promise = parse_multiplicative(scope)
                        expr_promise = promise_binop(
                            scope,
                            token,
                            op,
                            expr_promise,
                            right_promise,
                        )
                        break
                else:
                    break
            return expr_promise

        def parse_multiplicative(scope):
            expr_promise = parse_unary(scope)
            while True:
                token = peek()
                for op in ('*', '/', '%'):
                    if consume(op):
                        right_promise = parse_unary(scope)
                        expr_promise = promise_binop(
                            scope,
                            token,
                            op,
                            expr_promise,
                            right_promise,
                        )
                        break
                else:
                    break
            return expr_promise

        _binop_table = {
            '+': '__add',
            '-': '__sub',
            '*': '__mul',
            '/': '__div',
            '%': '__mod',
        }

        def promise_binop(scope, token, op, left_promise, right_promise):
            @Promise
            def promise():
                left = left_promise.resolve()
                right = right_promise.resolve()
                if op == '||':
                    assert left.type == IR.BOOL, left.type
                    assert right.type == IR.BOOL, right.type
                    return IR.LogicalOr(
                        token,
                        left,
                        right,
                    )
                if op == '&&':
                    assert left.type == IR.BOOL, left.type
                    assert right.type == IR.BOOL, right.type
                    return IR.LogicalAnd(
                        token,
                        left,
                        right,
                    )
                if (op in ('+', '-', '*', '/', '%') and
                        left.type in IR.NUMERIC_TYPES and
                        left.type == right.type):
                    return IR.PrimitiveBinop(
                        token,
                        left.type,
                        op,
                        left,
                        right,
                    )
                if (op in ('==', '<=', '>=', '<', '>', '!=') and
                        left.type in IR.NUMERIC_TYPES and
                        left.type == right.type):
                    return IR.PrimitiveBinop(
                        token,
                        IR.BOOL,
                        op,
                        left,
                        right,
                    )
                if (op in ('+', '-') and
                        isinstance(left.type, IR.PointerType) and
                        right.type == IR.INT):
                    if left.type == IR.VOIDP:
                        with scope.push(token):
                            raise scope.error(
                                f'Cannot do arithmetic on void pointers')
                    return IR.PointerAndIntArithmetic(
                        token,
                        left.type,
                        left,
                        op,
                        right,
                    )
                if (isinstance(left.type, IR.Retainable) and
                        isinstance(right.type, IR.Retainable)):
                    op_name = _binop_table[op]
                    return IR.InstanceMethodCall(
                        token,
                        op_name,
                        [
                            scope.convert(left, IR.VAR_TYPE),
                            scope.convert(right, IR.VAR_TYPE),
                        ],
                    )
                with scope.push(token):
                    key = (op, left.type, right.type)
                    raise scope.error(f'Unsupported binop {key}')
            return promise

        def promise_unop(scope, token, op, expr_promise):
            @Promise
            def promise():
                expr = expr_promise.resolve()
                if op in ('+', '-') and expr.type in IR.NUMERIC_TYPES:
                    return IR.PrimitiveUnop(
                        token,
                        expr.type,
                        op,
                        expr,
                    )
                if op == '!' and expr.type in IR.PRIMITIVE_TRUTHY_TYPES:
                    return IR.PrimitiveUnop(
                        token,
                        IR.BOOL,
                        op,
                        expr,
                    )
                with scope.push(token):
                    raise scope.error(f'Unsupported unary operation')
            return promise

        def parse_unary(scope):
            token = peek()
            for op in ('+', '-'):
                if consume(op):
                    expr_promise = parse_unary(scope)
                    return promise_unop(scope, token, op, expr_promise)
            return parse_postfix(scope)

        def parse_args(scope):
            argps = []
            expect('(')
            with skipping_newlines(True):
                while not consume(')'):
                    argps.append(parse_expression(scope))
                    if not consume(','):
                        expect(')')
                        break
            return Promise(lambda: [p.resolve() for p in argps])

        def promise_fcall(scope, token, function_promise, argsp):
            @Promise
            def promise():
                f = function_promise.resolve()
                raw_args = argsp.resolve()
                if isinstance(f.type, IR.FunctionType):
                    args = IR.convert_args(f.type, scope, token, raw_args)
                    return IR.FunctionCall(
                        token,
                        f.type.rtype,
                        f,
                        args,
                    )
                elif isinstance(f, IR.StaticMethodName):
                    args = IR.convert_args(f.defn.type, scope, token, raw_args)
                    return IR.StaticMethodCall(
                        token,
                        f.defn,
                        args,
                    )
                elif isinstance(f, IR.TraitOrClassName):
                    # If using a Class name like a function,
                    # the 'new' static method is implicitly called.
                    defn = get_static_method(scope, token, f.cls, 'new')
                    args = IR.convert_args(defn.type, scope, token, raw_args)
                    return IR.StaticMethodCall(
                        token,
                        defn,
                        args,
                    )
                elif isinstance(f, IR.InstanceMethodReference):
                    args = [f.owner] + [
                        scope.convert(arg, IR.VAR_TYPE) for arg in raw_args
                    ]
                    return IR.InstanceMethodCall(
                        token,
                        f.name,
                        args,
                    )
                else:
                    with scope.push(token):
                        raise scope.error(f'{f.type} is not a function')
            return promise

        def get_field_defn(scope, token, type, field_name):
            if not isinstance(type,
                    (IR.StructDefinition, IR.ClassDefinition)):
                with scope.push(token):
                    raise scope.error(f'{type} is not a struct or class type')
            fields = [f for f in type.fields if f.name == field_name]
            if not fields:
                with scope.push(token):
                    raise scope.error(
                        f'{field_name} is not a member of {type}')
            field, = fields
            return field

        def get_static_method(scope, token, cls, name):
            for method in cls.static_methods:
                if method.name == name:
                    return method
            with scope.push(token), scope.push(cls.token):
                raise scope.error(
                    f'No such static method with name {repr(name)}')

        def promise_get_field(scope, token, exprp, fname):
            @Promise
            def promise():
                expr = exprp.resolve()
                if isinstance(expr, IR.TraitOrClassName):
                    defn = get_static_method(scope, token, expr.cls, fname)
                    return IR.StaticMethodName(token, defn)
                elif isinstance(expr.type, IR.StructDefinition):
                    defn = get_field_defn(scope, token, expr.type, fname)
                    return IR.GetStructField(
                        token,
                        expr,
                        defn,
                    )
                elif isinstance(expr.type, IR.ClassDefinition) or (
                        expr.type == IR.VAR_TYPE):
                    return IR.InstanceMethodReference(
                        token,
                        scope.convert(expr, IR.VAR_TYPE),
                        fname,
                    )
                else:
                    with scope.push(token):
                        raise scope.error(
                            f'{expr.type} is not a struct or class type'
                        )
            return promise

        def promise_get_field_arrow(scope, token, exprp, fname):
            @Promise
            def promise():
                expr = exprp.resolve()
                if isinstance(expr.type, IR.ClassDefinition):
                    defn = get_field_defn(scope, token, expr.type, fname)
                    return IR.GetClassField(
                        token,
                        expr,
                        defn,
                    )
                else:
                    with scope.push(token):
                        raise scope.error(
                            f'{expr.type} is not a class type'
                        )
            return promise

        def promise_set_field(scope, token, exprp, fname, valp):
            @Promise
            def promise():
                expr = exprp.resolve()
                defn = get_field_defn(scope, token, expr.type, fname)
                val = scope.convert(valp.resolve(), defn.type)
                if isinstance(expr.type, IR.StructDefinition):
                    return IR.SetStructField(
                        token,
                        expr,
                        defn,
                        val,
                    )
                else:
                    with scope.push(token):
                        raise scope.error(
                            f'{expr.type} is not a struct or class type'
                        )
            return promise

        def promise_set_field_arrow(scope, token, exprp, fname, valp):
            @Promise
            def promise():
                expr = exprp.resolve()
                defn = get_field_defn(scope, token, expr.type, fname)
                val = scope.convert(valp.resolve(), defn.type)
                if isinstance(expr.type, IR.ClassDefinition):
                    return IR.SetClassField(
                        token,
                        defn,
                        expr,
                        val,
                    )
                else:
                    with scope.push(token):
                        raise scope.error(
                            f'{expr.type} is not a class type'
                        )
            return promise

        def process_assignment(scope, token, target, val):
            if isinstance(target, IR.LocalName):
                decl = target.decl
                if not decl.mutable:
                    with scope.push(token), scope.push(decl.token):
                        raise scope.error(
                            f'Tried to assign to non-mutable variable')
                return IR.SetLocalName(
                    token,
                    decl,
                    scope.convert(val, decl.type),
                )
            elif isinstance(target, IR.GetClassField):
                return IR.SetClassField(
                    token,
                    target.field_defn,
                    target.expr,
                    val,
                )
            elif isinstance(target, IR.GetStructField):
                return IR.SetStructField(
                    token,
                    target.expr,
                    target.field_defn,
                    val,
                )
            elif isinstance(target, IR.GlobalName):
                return IR.SetGlobalName(token, target.defn, val)
            with scope.push(token):
                raise scope.error(f'Not assignable')

        def parse_postfix(scope):
            expr = parse_primary(scope)
            while True:
                token = peek()
                if at('('):
                    argsp = parse_args(scope)
                    expr = promise_fcall(
                        scope=scope,
                        token=token,
                        function_promise=expr,
                        argsp=argsp,
                    )
                elif consume('.'):
                    name = expect_id()
                    expr = promise_get_field(scope, token, expr, name)
                elif consume('->'):
                    name = expect_id()
                    expr = promise_get_field_arrow(
                            scope, token, expr, name)
                elif consume('='):
                    valp = parse_expression(scope)
                    expr = pcall(process_assignment, scope, token, expr, valp)
                else:
                    break
            return expr

        def promise_name(scope, token, name):
            @Promise
            def promise():
                with scope.push(token):
                    defn = scope[name]
                if isinstance(defn, IR.FunctionDefinition):
                    return IR.FunctionName(token, defn)
                if isinstance(defn, IR.BaseLocalVariableDeclaration):
                    return IR.LocalName(token, defn)
                if isinstance(defn, IR.TraitOrClassDefinition):
                    return IR.TraitOrClassName(token, defn)
                if isinstance(defn, IR.FieldDefinition):
                    this_type = scope.get_this_type()
                    return IR.GetClassField(
                        token,
                        IR.This(token, this_type),
                        defn,
                    )
                if isinstance(defn, IR.DeleteQueueDeclaration):
                    return IR.DeleteQueueName(token)
                if isinstance(defn, IR.GlobalVariableDefinition):
                    return IR.GlobalName(token, defn)
                if isinstance(defn, IR.StaticMethodDefinition):
                    return IR.StaticMethodName(token, defn)
                if isinstance(defn, IR.InstanceMethodDefinition):
                    return IR.InstanceMethodReference(
                        token,
                        scope.convert(
                            promise_this(scope, token).resolve(),
                            IR.VAR_TYPE,
                        ),
                        defn.name,
                    )
                with scope.push(token):
                    raise scope.error(f'{defn} is not a variable')
            return promise

        def promise_throw_string_literal(scope, token, value):
            @Promise
            def promise():
                if scope['@ec'].extern:
                    with scope.push(token):
                        raise scope.error(
                            f'You cannot throw from an extern context')
                return IR.ThrowStringLiteral(token, value)
            return promise

        def promise_malloc(scope, token, type_promise):
            @Promise
            def promise():
                type = type_promise.resolve()
                if not isinstance(type, IR.ClassDefinition):
                    with scope.push(token):
                        raise scope.error(
                            f'Malloc ($) only allowed for Class types, '
                            f'but got {type}')
                if type.module_name != module_name:
                    with scope.push(token):
                        raise scope.error(
                            f'Malloc ($) only allowed from same file '
                            f'as definition of the class'
                        )
                return IR.Malloc(token, type)
            return promise

        def promise_this(scope, token):
            @Promise
            def promise():
                ec = scope['@ec']
                if ec.this_type is None:
                    with scope.push(token):
                        raise scope.error(f"'this' cannot be used here ")
                return IR.This(token, ec.this_type)
            return promise

        def make_cast(scope, token, type, arg):
            if ((isinstance(type, IR.PointerType) and
                    isinstance(arg.type, IR.PointerType) or
                    (type in IR.PRIMITIVE_TRUTHY_TYPES and
                        arg.type in IR.PRIMITIVE_TRUTHY_TYPES))):
                return IR.Cast(token, arg, type)
            with scope.push(token):
                raise scope.error(f'Unsupported cast to {type}')

        def parse_if(scope):
            token = expect('if')
            expect('(')
            with skipping_newlines(True):
                cond_promise = parse_truthy_expression(scope)
                expect(')')
            body_promise = parse_block(scope)
            if consume('else'):
                if at('if'):
                    other_promise = parse_if(scope)
                else:
                    other_promise = parse_block(scope)
            else:
                other_promise = Promise.value(IR.Block(token, [], []))
            @Promise
            def promise():
                cond = cond_promise.resolve()
                body = body_promise.resolve()
                other = other_promise.resolve()
                return IR.If(token, cond, body, other)
            return promise

        def parse_while(scope):
            token = expect('while')
            expect('(')
            with skipping_newlines(True):
                cond_promise = parse_truthy_expression(scope)
                expect(')')
            body_promise = parse_block(scope)
            @Promise
            def promise():
                cond = cond_promise.resolve()
                body = body_promise.resolve()
                return IR.While(token, cond, body)
            return promise

        def promise_string_literal(token, scope, value):
            @Promise
            def promise():
                c_str_expr = IR.CStringLiteral(token, value)
                if IR.STRING_CONVERSION_FUNCTION_NAME in scope:
                    return promise_fcall(
                        scope=scope,
                        token=token,
                        function_promise=promise_name(
                            scope,
                            token,
                            IR.STRING_CONVERSION_FUNCTION_NAME,
                        ),
                        argsp=Promise(lambda: [c_str_expr]),
                    ).resolve()
                else:
                    with scope.push(token):
                        raise scope.error(
                            f'Normal string literals are not available '
                            f'in this context, use c"..." style for '
                            f'C string literals')
                return expr
            return promise

        def promise_list_display(token, scope, element_promises):
            @Promise
            def promise():
                if (IR.NEW_LIST_FUNCTION_NAME not in scope or
                        IR.LIST_PUSH_FUNCTION_NAME not in scope):
                    with scope.push(token):
                        raise scope.error(
                            f'List displays not available in this context')
                expr = promise_fcall(
                    scope=scope,
                    token=token,
                    function_promise=promise_name(
                        scope,
                        token,
                        IR.NEW_LIST_FUNCTION_NAME,
                    ),
                    argsp=Promise(lambda: []),
                ).resolve()
                for element_promise in element_promises:
                    expr = promise_fcall(
                        scope=scope,
                        token=token,
                        function_promise=promise_name(
                            scope,
                            token,
                            IR.LIST_PUSH_FUNCTION_NAME,
                        ),
                        argsp=Promise(lambda: [
                            expr,
                            element_promise.resolve(),
                        ])
                    ).resolve()
                return expr
            return promise

        def parse_list_display(scope):
            token = expect('[')
            element_promises = []
            while not consume(']'):
                element_promises.append(parse_expression(scope))
                if not consume(','):
                    expect(']')
                    break
            return promise_list_display(token, scope, element_promises)

        def parse_primary(scope):
            token = peek()
            if at('if'):
                return parse_if(scope)
            if at('while'):
                return parse_while(scope)
            if consume('$'):
                if at('{'):
                    return parse_block(scope)
                else:
                    expect('(')
                    type_promise = parse_type(scope)
                    expect(')')
                    return promise_malloc(scope, token, type_promise)
            if consume('('):
                with skipping_newlines(True):
                    expr = parse_expression(scope)
                    expect(')')
                    return expr
            if at('['):
                return parse_list_display(scope)
            if consume('this'):
                return promise_this(scope, token)
            if consume('NAME'):
                name = token.value
                return promise_name(scope, token, name)
            if consume('INT'):
                return Promise.value(IR.IntLiteral(token, token.value))
            if consume('FLOAT'):
                return Promise.value(IR.FloatLiteral(token, token.value))
            if consume('C_STRING'):
                return Promise.value(IR.CStringLiteral(token, token.value))
            if consume('STRING'):
                return promise_string_literal(token, scope, token.value)
            if consume('throw'):
                # For now, only allow string literals
                value = expect('STRING').value
                return promise_throw_string_literal(scope, token, value)
            if consume('static_cast'):
                expect('(')
                with skipping_newlines(True):
                    type_promise = parse_type(scope)
                    expect(',')
                    arg_promise = parse_expression(scope)
                    expect(')')
                    return pcall(
                        make_cast,
                        scope,
                        token,
                        type_promise,
                        arg_promise,
                    )
            with scope.push(peek()):
                raise scope.error(f'Expected expression but got {peek()}')

        def check_fields(scope, fields, *, allow_retainable_fields):
            for field in fields:
                if not field.type.declarable:
                    with scope.push(field.token):
                        raise scope.error(
                            f'{field.type} is not declarable'
                        )
                elif (not allow_retainable_fields and
                        isinstance(field.type, IR.Retainable)):
                    with scope.push(field.token):
                        raise scope.error(
                            'Retainable types are not allowed here'
                        )

        def check_member_names(scope, members):
            table = dict()
            for field in members:
                if field.name in table:
                    token = field.token
                    other = table[field.name].token
                    with scope.push(token), scope.push(other):
                        raise scope.error(
                            f'Member name conflict for {field.name}')
                table[field.name] = field

        def parse_struct(scope, token, extern):
            name = expect_id()

            field_promises = []
            expect('{')
            consume_all('\n')
            while not consume('}'):
                field_token = peek()
                field_extern = bool(consume('extern'))
                field_type_promise = parse_type(scope)
                field_name = expect_id()
                field_promises.append(pcall(
                    IR.FieldDefinition,
                    field_token,
                    field_extern or extern,
                    field_type_promise,
                    field_name,
                ))
                expect('\n')
                consume_all('\n')

            defn = IR.StructDefinition(
                token,
                extern,
                module_name,
                name,
                field_promises,
            )
            scope[name] = defn

            @Promise
            def promise():
                fields = defn.fields
                check_member_names(scope, fields)
                check_fields(scope, fields, allow_retainable_fields=False)
                defn.declarable = True
                return defn

            return promise

        def parse_traits(scope):
            start_token = peek()
            if consume('('):
                token_and_names = []
                while not consume(')'):
                    token = peek()
                    token_and_names.append([token, expect_id()])
                    if not consume(','):
                        expect(')')
                        break
            else:
                token_and_names = None

            @Promise
            def promise():
                if token_and_names is not None:
                    traits = []
                    for token, name in token_and_names:
                        with scope.push(token):
                            trait = scope[name]
                            if not isinstance(trait, IR.TraitDefinition):
                                with scope.push(trait.token):
                                    raise scope.error(
                                        f'{name} is not a trait')
                            if not trait.defined:
                                with scope.push(trait.token):
                                    raise scope.error(
                                        f'Base traits must be defined '
                                        f'before any class or trait that '
                                        f'derives from it')
                        traits.append(trait)
                elif implicit_builtins:
                    # If 'builtins' is implicitly being pulled in,
                    root_trait = (
                        scope
                            .load_scope_for(BUILTINS_MODULE_NAME)
                            [ROOT_TRAIT_NAME]
                    )
                    if not isinstance(root_trait, IR.TraitDefinition):
                        with scope.push(root_trait.token):
                            raise scope.error(f'Root trait is not a trait!')
                    return [root_trait]
                else:
                    with scope.push(start_token):
                        raise scope.error(
                            f'If implicit builtins is disabled, '
                            f'an explicit traits list must be specified '
                            f'even if it is empty.'
                        )
                return traits

            return promise

        def parse_class(is_trait, outer_scope, token):
            name = expect_id()
            traits_promise = parse_traits(outer_scope)

            field_promises = None if is_trait else []
            static_method_promises = []
            instance_method_promises = []

            if is_trait:
                defn = IR.TraitDefinition(
                    token=token,
                    module_name=module_name,
                    short_name=name,
                    static_method_promises=static_method_promises,
                    instance_method_promises=instance_method_promises,
                    traits_promise=traits_promise,
                )
            else:
                defn = IR.ClassDefinition(
                    token=token,
                    extern=False,
                    module_name=module_name,
                    short_name=name,
                    field_promises=field_promises,
                    delete_hook_promise=Promise(lambda: delete_hook_ptr[0]),
                    static_method_promises=static_method_promises,
                    instance_method_promises=instance_method_promises,
                    traits_promise=traits_promise,
                )
            outer_scope[name] = defn

            class_scope = Scope(outer_scope)

            delete_hook_ptr = [None]
            expect('{')
            consume_all('\n')
            while not consume('}'):
                member_token = peek()
                if not is_trait and consume('delete'):
                    expect_delete_hook(
                        scope=class_scope,
                        token=member_token,
                        delete_hook_ptr=delete_hook_ptr,
                        cls=defn,
                    )
                elif consume('static'):
                    static_method_promises.append(parse_static_method(
                        class_scope, member_token, defn,
                    ))
                else:
                    member_type_promise = parse_type(class_scope)
                    member_name = expect_id()
                    if is_trait or at('('):
                        instance_method_promises.append(
                            parse_instance_method(
                                class_scope,
                                member_token,
                                defn,
                                member_type_promise,
                                member_name,
                            )
                        )
                    else:
                        field_promise = pcall(
                            IR.FieldDefinition,
                            member_token,
                            False,
                            member_type_promise,
                            member_name,
                        )
                        field_promises.append(field_promise)
                        class_scope.set_promise(
                            member_token,
                            member_name,
                            field_promise,
                        )
                        expect('\n')
                consume_all('\n')

            if delete_hook_ptr[0] is None:
                delete_hook_ptr[0] = IR.DeleteHook(
                    token,
                    Promise.value(IR.Block(token, [], [])),
                )

            @Promise
            def promise():
                if not is_trait:
                    fields = defn.fields
                    # We don't need to call 'check_member_names' because
                    # if there is a member name conflict,
                    # class_scope.set_promise would've thrown something.
                    check_fields(
                        outer_scope,
                        fields,
                        allow_retainable_fields=True,
                    )
                defn.traits  # verify all traits are defined by now
                defn.defined = True
                return defn

            return promise

        def parse_instance_method(
                class_scope, token, cls, rtype_promise, name):
            method_scope = Scope(class_scope)
            plist_promise = parse_param_list(method_scope)
            method_scope['@ec'] = IR.ExpressionContext(
                token=token,
                extern=False,
                this_type=(
                    IR.VAR_TYPE
                    if isinstance(cls, IR.TraitDefinition) else
                    cls
                ),
            )
            body_promise = parse_block(method_scope)
            defn_promise = Promise.value(IR.InstanceMethodDefinition(
                token,
                Promise.value(cls),
                rtype_promise,
                name,
                plist_promise,
                promise_block(
                    token=token,
                    scope=method_scope,
                    decl_promises=[],
                    expr_promises=[
                        Promise(lambda: method_scope.convert(
                            body_promise.resolve(),
                            rtype_promise.resolve(),
                        )),
                    ],
                ),
            ))
            class_scope.set_promise(token, name, defn_promise)
            return defn_promise

        def parse_static_method(class_scope, token, cls):
            rtype_promise = parse_type(class_scope)
            method_scope = Scope(class_scope)
            name = expect_id()
            plist_promise = parse_param_list(method_scope)
            method_scope['@ec'] = IR.ExpressionContext(
                token=token,
                extern=False,
                this_type=None,
            )
            body_promise = parse_block(method_scope)
            defn_promise = Promise.value(IR.StaticMethodDefinition(
                token,
                Promise.value(cls),
                rtype_promise,
                name,
                plist_promise,
                body_promise,
            ))
            class_scope.set_promise(token, name, defn_promise)
            return defn_promise

        def expect_delete_hook(scope, token, delete_hook_ptr, cls):
            if delete_hook_ptr[0]:
                dtok = delete_hook_ptr[0].token
                with scope.push(dtok), scope.push(token):
                    raise scope.error('Duplicate delete definition')
            delete_scope = Scope(scope)
            delete_scope['@ec'] = IR.ExpressionContext(
                token,
                extern=True,
                this_type=cls,
            )
            delete_scope['__delete_queue'] = IR.DeleteQueueDeclaration(token)
            delete_hook_ptr[0] = IR.DeleteHook(
                token,
                parse_block(delete_scope),
            )
            return delete_hook_ptr[0]

        def parse_function(outer_scope, token, extern, type_promise, name):
            func_scope = Scope(outer_scope)
            plist_promise = parse_param_list(func_scope)
            func_scope['@ec'] = IR.ExpressionContext(
                token=token,
                extern=extern,
                this_type=None,
            )
            has_body = not consume('\n')
            raw_body_promise = (
                parse_block(func_scope) if has_body else None
            )
            @Promise
            def body_promise():
                defn = defn_promise.resolve()
                if has_body:
                    raw_body = raw_body_promise.resolve()
                    assert raw_body is not None
                    if defn.rtype != IR.VOID:
                        body = promise_block(
                            token=token,
                            scope=func_scope,
                            decl_promises=[],
                            expr_promises=[Promise.value(
                                outer_scope.convert(raw_body, defn.rtype)
                            )],
                        ).resolve()
                    else:
                        body = raw_body
                    return body
                else:
                    return None
            defn_promise = Promise(lambda: IR.FunctionDefinition(
                token,
                extern,
                type_promise.resolve(),
                module_name,
                name,
                plist_promise.resolve(),
                has_body,
                body_promise,
            ))
            outer_scope.set_promise(token, name, defn_promise)
            return defn_promise

        def parse_global_var_defn(
                outer_scope, token, extern, type_promise, name):
            assert not extern, 'extern global variables not yet supported'
            expect('=')
            expr_scope = Scope(outer_scope)
            expr_scope['@ec'] = IR.ExpressionContext(
                token=token,
                extern=False,
                this_type=None,
            )
            expr_promise = parse_expression(expr_scope)
            defn_promise = pcall(
                IR.GlobalVariableDefinition,
                token,
                extern,
                type_promise,
                module_name,
                name,
                expr_promise.map(lambda expr:
                    expr_scope.convert(expr, type_promise.resolve())
                ),
            )
            outer_scope.set_promise(token, name, defn_promise)
            return defn_promise

        def parse_global(scope):
            token = peek()
            extern = bool(consume('extern'))
            if consume('typedef'):
                expect('*')
                name = expect_id()
                scope[name] = defn = IR.PrimitiveTypeDefinition(token, name)
                return Promise.value(defn)
            elif consume('struct'):
                return parse_struct(scope, token, extern)
            elif at('class') or at('trait'):
                if extern:
                    with scope.push(token):
                        raise scope.error(
                            f'Classes and traits cannot be extern')
                is_trait = bool(consume('trait'))
                if not is_trait:
                    expect('class')
                return parse_class(is_trait, scope, token)
            else:
                type_promise = parse_type(scope)
                name = expect_id()
                if at('('):
                    return parse_function(
                        scope, token, extern, type_promise, name)
                else:
                    return parse_global_var_defn(
                        scope, token, extern, type_promise, name)

        module_token = peek()
        includes = []
        importps = []
        defnps = []

        implicit_builtins = True
        consume_all('\n')
        while at('#'):
            itoken = expect('#')
            if at('STRING') and peek().value == 'no builtins':
                expect('STRING')
                implicit_builtins = False
            else:
                expect_name('include')
                use_quotes = at('STRING')
                if use_quotes:
                    ivalue = expect('STRING').value
                else:
                    expect('<')
                    parts = []
                    while not consume('>'):
                        parts.append(gettok().value)
                    ivalue = ''.join(parts)
                includes.append(IR.Include(itoken, use_quotes, ivalue))
            consume_all('\n')

        if implicit_builtins:
            for export_name in _builtins_export_names:
                importps.append(module_scope.from_import(
                    token=builtin_token,
                    module_name=BUILTINS_MODULE_NAME,
                    exported_name=export_name,
                    alias=export_name,
                ))

        while at('from'):
            importps.append(parse_import(module_scope))
            consume_all('\n')
        while not at('EOF'):
            defnps.append(parse_global(module_scope))
            consume_all('\n')

        return module_scope, Promise(lambda: IR.Module(
            module_token,
            module_name,
            includes,
            [p.resolve() for p in importps],
            [p.resolve() for p in defnps],
        ))


@Namespace
def C(ns):
    THIS_NAME = 'KLC_this'
    ENCODED_FUNCTION_PREFIX = 'KLCFN'
    ENCODED_GLOBAL_VARIABLE_GETTER_PREFIX = 'KLCGVG'
    ENCODED_GLOBAL_VARIABLE_INITVAR_PREFIX = 'KLCGVI'
    ENCODED_GLOBAL_VARIABLE_INITIALIZER_PREFIX = 'KLCGVR'
    ENCODED_GLOBAL_VARIABLE_NAME_PREFIX = 'KLCGVN'
    ENCODED_LOCAL_PARAM_PREFIX = 'KLCLP'
    ENCODED_LOCAL_VARIABLE_PREFIX = 'KLCLV'
    ENCODED_STRUCT_PREFIX = 'KLCST'
    ENCODED_STRUCT_FIELD_PREFIX = 'KLCSF'
    ENCODED_CLASS_DESCRIPTOR_PREFIX = 'KLCCP'
    ENCODED_CLASS_MALLOC_PREFIX = 'KLCCM'
    ENCODED_CLASS_DELETE_HOOK_PREFIX = 'KLCDH'
    ENCODED_CLASS_DELETER_PREFIX = 'KLCCD'
    ENCODED_CLASS_STRUCT_PREFIX = 'KLCCS'
    ENCODED_CLASS_FIELD_PREFIX = 'KLCCF'
    ENCODED_METHOD_LIST_PREFIX = 'KLCML'
    ENCODED_STATIC_METHOD_PREFIX = 'KLCSM'
    ENCODED_INSTANCE_METHOD_PREFIX = 'KLCIM'
    OUTPUT_PTR_NAME = 'KLC_output_ptr'
    CLASS_HEADER_FIELD_NAME = 'header'
    METHOD_PARAM_ARGC_NAME = 'KLC_argc'
    METHOD_PARAM_ARGV_NAME = 'KLC_argv'
    HEADER_STRUCT_NAME = 'KLC_Header'
    ERROR_POINTER_NAME = 'KLC_error'
    ERROR_POINTER_TYPE = IR.ERROR_POINTER_TYPE
    HEADER_POINTER_TYPE = IR.HEADER_POINTER_TYPE
    STACK_POINTER_TYPE = IR.STACK_POINTER_TYPE
    STACK_POINTER_NAME = 'KLC_stack'
    DEBUG_FUNC_NAME_NAME = 'KLC_debug_func_name'
    DEBUG_FILE_NAME_NAME = 'KLC_debug_file_name'
    CALL_METHOD_FUNCTION_NAME = 'KLC_call_method'
    DELETE_QUEUE_NAME = 'KLC_delete_queue'
    DELETER_TYPE = IR.FunctionType(
        extern=True,
        rtype=IR.VOID,
        paramtypes=[
            HEADER_POINTER_TYPE,
            IR.PointerType(HEADER_POINTER_TYPE),
        ],
        vararg=False,
    )
    METHOD_TYPE = IR.FunctionType(
        extern=False,
        rtype=IR.VAR_TYPE,
        paramtypes=[
            IR.c_int,
            IR.PointerType(IR.VAR_TYPE),
        ],
        vararg=False,
    )

    encode_map = {
        '_': '_U',  # U for Underscore
                    # underscore si the escape cahracter

        '.': '_D',  # D for Dot
                    # For use in module names

        '#': '_H',  # H for Hashtag
                    # For generating method names

        '%': '_M',  # M for Modulo
                    # For auto-generated symbol names,
                    # e.g. any temp variables,
                    # the '%str' function that automatically
                    # converts C string literals into
                    # String when builtin is included.
    }

    def encode_char(c):
        if c.isalnum() or c.isdigit():
            return c
        elif c in encode_map:
            return encode_map[c]
        raise TypeError(f'Invalid name character {c}')

    def encode(name, prefix):
        return prefix + ''.join(map(encode_char, name))

    def relative_header_path_from_name(module_name):
        return module_name.replace('.', os.path.sep) + '.k.h'

    def relative_source_path_from_name(module_name):
        return module_name.replace('.', os.path.sep) + '.k.c'

    class TranslationUnit(IR.Node):
        node_fields = (
            ('name', str),
            ('h', str),
            ('c', str),
        )

    class Context:
        """Translation Context

        Memory Management:
            The basic approach is with automatic reference counting.
            To make the reference counting logic here less error prone,
            we use a combination of three methods:

                declare(type, [cname]), and
                retain_scope()
                release_all()

                declare(type, [cname]) -> cname:
                    Declares a variable in the current retain scope
                    that will automatically be released when
                    the current retain_scope exits.
                retain_scope() (Contxt manager)
                    Used with 'with', creates a scope over which all
                    variables declared within that scope are retained.
                release_all()
                    Emits code that will release all retained variables
                    in current scope.
                    The normal code for releasing retained variables
                    at the end of the scope will still be emitted.
                    This is useful for e.g. early returns

        FractalStringBuilder
            The text is filled with FractalStringBuilder which has
            the ability to bookmark locations for filling
            in content even after content after it has been filled.
            This is really useful in some cases (e.g. with declaring
            variables in the beginning of blocks, and for headers
            where forward declares must come before other declares, etc.),
            however, this feature should not be needed very much
            during the main code generation of expressions.
            Manually spawning the builder can cause issues with
            some Context methods here, like with the memory management,
            or with indentations.
        """

        def __init__(self):
            self.out = FractalStringBuilder(0)
            self.includes = self.out.spawn()
            self.static_fdecls = self.out.spawn()
            self.static_vars = self.out.spawn()
            self.next_temp_var_id = 0

            # Map of declared C local variables to their types
            # These are updated as we enter and exit scopes.
            self._scope = collections.OrderedDict()

            # The label to jump to to exit the current scope
            # (e.g. for exceptions or early returns)
            # We can't willy-nilly exit the function because we still
            # need to release all the variables.
            # Jumping to the designated label ensures variables in
            # all scopes are properly released.
            self._exit_jump_label = None

            # A place where temporary variables may be declared
            # We need this mechanism because in C89, you can't
            # willy-nilly declare variables anywhere (they
            # have to be declared at the beginning of a block).
            self._decls = None

        def _new_unique_cname(self):
            name = f'KLCT{self.next_temp_var_id}'
            self.next_temp_var_id += 1
            return name

        def declare(self, t: IR.Type, cname=None):
            cname = (
                self._new_unique_cname() if cname is None else
                cname
            )
            assert cname not in self._scope, cname
            self._scope[cname] = t
            self._decls += f'{declare(t, cname)} = {init_expr_for(t)};'
            return cname

        @contextlib.contextmanager
        def retain_scope(self):
            old_decls = self._decls
            old_out = self.out
            old_exit_jump_label = self._exit_jump_label
            old_scope_size = len(self._scope)

            old_out += '{'
            self._decls = old_out.spawn(1)
            self.out = old_out.spawn(1)
            after_release = old_out.spawn(1)
            old_out += '}'

            new_exit_jump_label = self._new_unique_cname()
            self._exit_jump_label = new_exit_jump_label

            try:
                # We yield 'after_release' so that the caller may
                # add statements after all local variables have
                # been released
                yield after_release
            finally:
                # This is where all code exiting this block will jump
                # to before they exit.
                self.out += f'{new_exit_jump_label}:;'

                # Release all variables in this scope
                while len(self._scope) > old_scope_size:
                    cname, type = self._scope.popitem(last=True)
                    self.out += _release(type=type, cname=cname)

                # If there is a surrounding scope and an exception
                # is being thrown, we want to jump out of the outer
                # scope as well.
                if old_exit_jump_label:
                    self.out += f'if ({ERROR_POINTER_NAME}) ' '{'
                    with self.push_indent(1):
                        self.out += f'goto {old_exit_jump_label};'
                    self.out += '}'

                self._decls = old_decls
                self.out = old_out
                self._exit_jump_label = old_exit_jump_label

        def jump_on_error(self):
            self.out += f'if ({ERROR_POINTER_NAME}) ' '{'
            with self.push_indent(1):
                self.jump_out_of_scope()
            self.out += '}'

        def jump_out_of_scope(self):
            self.out += f'goto {self._exit_jump_label};'

        def retain(self, cname, out=None):
            assert cname is None or cname in self._scope, cname
            if cname is not None:
                out = out or self.out
                out += _retain(self._scope[cname], cname)

        def release(self, cname, out=None, clear_var=False):
            """Releases the given (declared) local variable.

            If clear_var is true, the given variable will also
            be set to NULL after release.

            If clear_var is not set, the given variable will
            be released naturally at the end of the scope.
            """
            assert cname is None or cname in self._scope, cname
            if cname is not None:
                out = out or self.out
                cmd = _release(self._scope[cname], cname)
                if cmd:
                    out += cmd
                    if clear_var:
                        out += f'{cname} = NULL;'

        @contextlib.contextmanager
        def push_indent(self, depth):
            old_out = self.out
            self.out = old_out.spawn(depth)
            try:
                yield
            finally:
                self.out = old_out

        def set_error_pointer(self, message):
            escaped = escape_str(message)
            self.out += (
                f'{ERROR_POINTER_NAME} = KLC_new_error_with_message('
                f'{STACK_POINTER_NAME}, "{escaped}");'
            )

    def _retain(type, cname):
        if isinstance(type, IR.ClassDefinition):
            return f'KLC_retain((KLC_Header*) {cname});'
        elif type == IR.VAR_TYPE:
            return f'KLC_retain_var({cname});'
        else:
            assert not isinstance(type, IR.Retainable), type
            return ''

    def _release(type, cname):
        if isinstance(type, IR.ClassDefinition):
            return f'KLC_release((KLC_Header*) {cname});'
        elif type == IR.VAR_TYPE:
            return f'KLC_release_var({cname});'
        else:
            assert not isinstance(type, IR.Retainable), type
            return ''

    def _release_on_exit(type, cname):
        if isinstance(type, IR.ClassDefinition):
            return f'KLC_release_on_exit((KLC_Header*) {cname});'
        elif type == IR.VAR_TYPE:
            return f'KLC_release_var_on_exit({cname});'
        else:
            assert not isinstance(type, IR.Retainable), type
            return ''

    @ns
    def write_out(tu: TranslationUnit, out_dir: str):
        header_path = os.path.join(
            out_dir,
            relative_header_path_from_name(tu.name),
        )
        source_path = os.path.join(
            out_dir,
            relative_source_path_from_name(tu.name),
        )
        os.makedirs(out_dir, exist_ok=True)
        with open(header_path, 'w') as f:
            f.write(tu.h)
        with open(source_path, 'w') as f:
            f.write(tu.c)

    @ns
    def translate(module: IR.Module) -> TranslationUnit:
        return TranslationUnit(
            module.token,
            module.name,
            translate_header(module),
            translate_source(module),
        )

    def _get_includes(module):
        incs = ['#include "kcrt.h"']

        # explicitly listed includes
        for inc in module.includes:
            if inc.use_quotes:
                incs.append(f'#include "{inc.value}"')
            else:
                incs.append(f'#include <{inc.value}>')

        # includes implied from imports
        for imp in module.imports:
            path = relative_header_path_from_name(imp.module_name)
            incs.append(f'#include "{path}"')

        # remove duplicates
        return tuple(collections.OrderedDict.fromkeys(incs))

    def translate_header(module: IR.Module) -> str:
        guard_name = (
            relative_header_path_from_name(module.name)
                .replace('.', '_')
                .replace(os.path.sep, '_')
        )
        msb = FractalStringBuilder(0)
        msb += f'#ifndef {guard_name}'
        msb += f'#define {guard_name}'
        sb = msb.spawn()
        msb += f'#endif/*{guard_name}*/'

        for include_line in _get_includes(module):
            sb += include_line

        fwdstruct = sb.spawn()
        structs = sb.spawn()
        gvardecls = sb.spawn()
        fdecls = sb.spawn()

        for defn in module.definitions:
            if isinstance(defn, IR.FunctionDefinition):
                fdecls += f'extern {proto_for(defn)};'
            elif isinstance(defn, IR.PrimitiveTypeDefinition):
                # These definitions are just to help the compiler,
                # nothing actually needs to get emitted for this.
                pass
            elif isinstance(defn, IR.StructDefinition):
                cname = get_c_struct_name(defn)
                # We only actually define the struct itself if
                #   1. the struct was not declard extern, and
                #   2. a body was explicitly provided.
                # If there is no body, this might not even be
                # a struct.
                if defn.fields is not None and not defn.extern:
                    fwdstruct += f'typedef struct {cname} {cname};'
                    structs += f'struct {cname} ' '{'
                    fields_out = structs.spawn(1)
                    for field in defn.fields:
                        fields_out += declare(
                            field.type,
                            get_c_struct_field_name(field),
                        ) + ';';
                    if not defn.fields:
                        # Empty structs aren't standard C.
                        # Add a dummy field.
                        fields_out += f'char dummy;'
                    structs += '};'
            elif isinstance(defn, IR.TraitOrClassDefinition):
                assert not defn.extern, defn

                if isinstance(defn, IR.ClassDefinition):
                    struct_name = get_class_struct_name(defn)

                    fwdstruct += (
                        f'typedef struct {struct_name} {struct_name};'
                    )

                    structs += f'struct {struct_name} ' '{'
                    fields_out = structs.spawn(1)
                    fields_out += (
                        f'{HEADER_STRUCT_NAME} {CLASS_HEADER_FIELD_NAME};'
                    )
                    for field in defn.fields:
                        fields_out += declare(
                            field.type,
                            get_class_c_struct_field_name(field),
                        ) + ';'
                    structs += '};'

                descriptor_name = get_class_descriptor_name(defn)
                gvardecls += f'extern KLC_Class {descriptor_name};'

                for static_method in defn.static_methods:
                    fdecls += f'extern {proto_for(static_method)};'

                for instance_method in defn.instance_methods:
                    fdecls += f'extern {proto_for(instance_method)};'
            elif isinstance(defn, IR.GlobalVariableDefinition):
                assert not defn.extern, defn
                getter_proto = get_global_var_getter_proto(defn)
                fdecls += f'extern {getter_proto};'
            else:
                raise Fubar([], defn)

        return str(msb)

    def translate_source(module: IR.Module) -> str:
        ctx = Context()
        ctx.includes += (
            f'#include "{relative_header_path_from_name(module.name)}"'
        )
        ctx.static_vars += (
            f'static const char* {DEBUG_FILE_NAME_NAME} = '
            f'"{module.token.source.filename}";'
        )
        for defn in module.definitions:
            D(defn, ctx)
        return str(ctx.out)

    def qualify(module_name, name, prefix):
        return (
            encode(name, prefix=prefix) if module_name is None else
            encode(f'{module_name}.{name}', prefix=prefix)
        )

    def get_global_var_getter_proto(defn: IR.GlobalVariableDefinition):
        getter_name = get_global_var_getter_name(defn)
        getter_type = get_global_var_getter_type(defn)
        return declare(
            getter_type,
            getter_name,
            [STACK_POINTER_NAME, OUTPUT_PTR_NAME],
        )

    def get_global_var_getter_name(defn: IR.GlobalVariableDefinition):
        assert isinstance(defn, IR.GlobalVariableDefinition), defn
        return qualify(
            defn.module_name,
            defn.short_name,
            prefix=ENCODED_GLOBAL_VARIABLE_GETTER_PREFIX,
        )

    def get_global_var_getter_type(defn: IR.GlobalVariableDefinition):
        assert isinstance(defn, IR.GlobalVariableDefinition), defn
        return IR.FunctionType(
            extern=False,
            rtype=defn.type,
            paramtypes=[],
            vararg=False,
        )

    def get_global_var_initializer_type(defn: IR.GlobalVariableDefinition):
        return get_global_var_getter_type(defn)

    def get_global_initvar_name(defn: IR.GlobalVariableDefinition):
        assert isinstance(defn, IR.GlobalVariableDefinition), defn
        return qualify(
            defn.module_name,
            defn.short_name,
            prefix=ENCODED_GLOBAL_VARIABLE_INITVAR_PREFIX,
        )

    def get_global_var_initializer_name(defn: IR.GlobalVariableDefinition):
        assert isinstance(defn, IR.GlobalVariableDefinition), defn
        return qualify(
            defn.module_name,
            defn.short_name,
            prefix=ENCODED_GLOBAL_VARIABLE_INITIALIZER_PREFIX,
        )

    def get_global_var_initializer_proto(defn: IR.GlobalVariableDefinition):
        assert isinstance(defn, IR.GlobalVariableDefinition), defn
        return declare(
            get_global_var_initializer_type(defn),
            get_global_var_initializer_name(defn),
            [STACK_POINTER_NAME, OUTPUT_PTR_NAME],
        )

    def get_global_var_name(defn: IR.GlobalVariableDefinition):
        return qualify(
            defn.module_name,
            defn.short_name,
            prefix=ENCODED_GLOBAL_VARIABLE_NAME_PREFIX,
        )

    def get_c_struct_name(defn: IR.StructDefinition):
        assert isinstance(defn, IR.StructDefinition), defn
        return (
            defn.short_name if defn.extern else
            qualify(
                defn.module_name,
                defn.short_name,
                prefix=ENCODED_STRUCT_PREFIX,
            )
        )

    def get_c_struct_field_name(defn: IR.FieldDefinition):
        return (
            defn.name if defn.extern else
            encode(defn.name, prefix=ENCODED_STRUCT_FIELD_PREFIX)
        )

    def get_class_descriptor_name(decl):
        assert isinstance(decl, IR.TraitOrClassDefinition), decl
        return qualify(
            decl.module_name,
            decl.short_name,
            prefix=ENCODED_CLASS_DESCRIPTOR_PREFIX,
        )

    def get_class_malloc_type(decl: IR.ClassDefinition):
        assert isinstance(decl, IR.ClassDefinition), decl
        return IR.FunctionType(
            extern=True,
            rtype=decl,
            paramtypes=[],
            vararg=False,
        )

    def get_class_malloc_name(decl: IR.ClassDefinition):
        assert isinstance(decl, IR.ClassDefinition), decl
        return qualify(
            decl.module_name,
            decl.short_name,
            prefix=ENCODED_CLASS_MALLOC_PREFIX,
        )

    def get_delete_hook_type(decl: IR.ClassDefinition):
        assert isinstance(decl, IR.ClassDefinition), decl
        return IR.FunctionType(
            extern=True,
            rtype=IR.VOID,
            paramtypes=[decl, IR.VOIDP],
            vararg=False,
        )

    def get_class_delete_hook_name(decl: IR.ClassDefinition):
        assert isinstance(decl, IR.ClassDefinition), decl
        return qualify(
            decl.module_name,
            decl.short_name,
            prefix=ENCODED_CLASS_DELETE_HOOK_PREFIX,
        )

    def get_class_deleter_name(decl: IR.ClassDefinition):
        assert isinstance(decl, IR.ClassDefinition), decl
        return qualify(
            decl.module_name,
            decl.short_name,
            prefix=ENCODED_CLASS_DELETER_PREFIX,
        )

    def get_class_struct_name(decl: IR.ClassDefinition):
        assert isinstance(decl, IR.ClassDefinition), decl
        assert not decl.extern, decl
        return qualify(
            decl.module_name,
            decl.short_name,
            prefix=ENCODED_CLASS_STRUCT_PREFIX,
        )

    def get_method_list_name(defn: IR.ClassDefinition):
        return qualify(
            defn.module_name,
            defn.short_name,
            prefix=ENCODED_METHOD_LIST_PREFIX,
        )

    def get_class_c_struct_field_name(defn: IR.FieldDefinition):
        assert not defn.extern, defn
        return encode(defn.name, prefix=ENCODED_CLASS_FIELD_PREFIX)

    def get_static_method_name(defn: IR.StaticMethodDefinition):
        assert isinstance(defn, IR.StaticMethodDefinition), defn
        return encode(
            f'{defn.cls.module_name}#{defn.cls.short_name}#{defn.name}',
            prefix=ENCODED_STATIC_METHOD_PREFIX,
        )

    def get_instance_method_name(defn: IR.InstanceMethodDefinition):
        assert isinstance(defn, IR.InstanceMethodDefinition), defn
        return encode(
            f'{defn.cls.module_name}#{defn.cls.short_name}#{defn.name}',
            prefix=ENCODED_INSTANCE_METHOD_PREFIX,
        )

    def encode_param_names(paramlist: IR.ParameterList):
        assert isinstance(paramlist, IR.ParameterList), paramlist
        return [
            encode(p.name, prefix=ENCODED_LOCAL_PARAM_PREFIX)
            for p in paramlist.params
        ]

    # Get the C name of variable with given declaration
    cvarname = Multimethod('cvarname')

    @cvarname.on(IR.Parameter)
    def cvarname(self):
        # NOTE: It isn't a typo that we use ENCODED_LOCAL_VARIABLE_PREFIX
        # here instead of ENCODED_LOCAL_PARAM_PREFIX.
        # At the beginning of every function body, we copy over all
        # parameters to local variables to avoid any possible
        # retain/release issues related to mutating parameter values.
        return encode(self.name, prefix=ENCODED_LOCAL_VARIABLE_PREFIX)

    @cvarname.on(IR.LocalVariableDeclaration)
    def cvarname(self):
        return encode(self.name, prefix=ENCODED_LOCAL_VARIABLE_PREFIX)

    @cvarname.on(IR.FunctionDefinition)
    def cvarname(self):
        return (
            self.short_name if self.extern else
            qualify(
                self.module_name,
                self.short_name,
                prefix=ENCODED_FUNCTION_PREFIX,
            )
        )

    @cvarname.on(IR.StaticMethodDefinition)
    def cvarname(self):
        return get_static_method_name(self)

    @cvarname.on(IR.InstanceMethodDefinition)
    def cvarname(self):
        return get_instance_method_name(self)

    declare = Multimethod('declare')

    @declare.on(IR.PrimitiveTypeDefinition)
    def declare(self, name):
        return f'{self.name} {name}'.strip()

    @declare.on(IR.StructDefinition)
    def declare(self, name):
        struct_name = get_c_struct_name(self)
        return f'{struct_name} {name}'.strip()

    @declare.on(IR.ClassDefinition)
    def declare(self, name):
        struct_name = get_class_struct_name(self)
        return f'{struct_name}* {name}'.strip()

    @declare.on(type(IR.VAR_TYPE))
    def declare(self, name):
        return f'KLC_var {name}'.strip()

    @declare.on(IR.PointerType)
    def declare(self, name):
        return declare(self.base, f' *{name}'.strip())

    @declare.on(IR.ConstType)
    def declare(self, name):
        return declare(self.base, f'const {name}'.strip())

    @declare.on(IR.FunctionType)
    def declare(self, name, pnames=None):
        # NOTE: For noraml (non-extern) functions,
        # the function signature is different from
        # the normal C signature you might expect.
        # In such cases, we always return void*
        # (indicate error), and use the first argument
        # for return type.
        if self.extern:
            return declare_raw_c_functype(self, name, pnames)
        else:
            rtype = ERROR_POINTER_TYPE
            pnames = None if pnames is None else (
                [STACK_POINTER_NAME, OUTPUT_PTR_NAME] + list(pnames)
            )
            paramtypes = [
                STACK_POINTER_TYPE,
                IR.PointerType(self.rtype)
            ] + list(self.paramtypes)
            return declare_raw_c_functype(
                IR.FunctionType(
                    extern=self.extern,
                    rtype=rtype,
                    paramtypes=paramtypes,
                    vararg=self.vararg,
                ),
                name=name,
                pnames=pnames,
            )

    def declare_raw_c_functype(self: IR.FunctionType, name, pnames=None):
        assert isinstance(self, IR.FunctionType), self

        if pnames is None:
            pnames = [''] * len(self.paramtypes)

        # take care with operator precedence.
        # we need extra parens here because function call
        # binds tighter than the pointer type modifier.
        if name.startswith('*'):
            name = f'({name})'

        params = ', '.join(
            [declare(ptype, pname)
                for ptype, pname in zip(self.paramtypes, pnames)] +
            (['...'] if self.vararg else []),
        )

        return declare(self.rtype, f'{name}({params})')

    # translate definition for source
    D = Multimethod('D')

    @D.on(IR.PrimitiveTypeDefinition)
    def D(self, ctx):
        # These definitions are just to help the compiler,
        # nothing actually needs to get emitted for this.
        pass

    def emit_function_body(
            *,
            ctx,
            debug_name,
            proto,
            extern,
            rtype,
            start_callback,
            body):
        ctx.out += proto
        with ctx.retain_scope() as after_release:
            ctx.out += (
                f'static const char* {DEBUG_FUNC_NAME_NAME} = '
                f'"{debug_name}";'
            )
            ctx.declare(ERROR_POINTER_TYPE, ERROR_POINTER_NAME)
            if extern:
                ctx.declare(STACK_POINTER_TYPE, STACK_POINTER_NAME)
                ctx.out += f'{STACK_POINTER_NAME} = KLC_new_stack();'

            # we have a callback here to allow callers to
            # customize the way that they process arguments
            start_callback(ctx)

            retvar = E(body, ctx)
            if rtype != IR.VOID:
                ctx.retain(retvar)

            if extern:
                after_release += f'KLC_delete_stack({STACK_POINTER_NAME});'
                after_release += f'if ({ERROR_POINTER_NAME}) ' '{'
                after_release += (
                    f'  KLC_panic_with_error({ERROR_POINTER_NAME});'
                )
                after_release += '}'
                if rtype != IR.VOID:
                    after_release += f'return {retvar};'
            else:
                if rtype != IR.VOID:
                    after_release += f'*{OUTPUT_PTR_NAME} = {retvar};'
                after_release += f'return {ERROR_POINTER_NAME};'

    def copy_params_to_locals(ctx, paramtypes, c_param_names, c_local_names):
        assert len(paramtypes) == len(c_local_names), c_local_names
        assert len(c_param_names) == len(c_local_names), c_local_names
        # We copy over all the parameters to local variables
        # so that we can treat them like normal local variables.
        for param_type, param_name, local_name in zip(
                paramtypes, c_param_names, c_local_names):
            ctx.declare(param_type, local_name)
            ctx.out += f'{local_name} = {param_name};'
            ctx.retain(local_name)

    get_global_defn_name = Multimethod('get_global_defn_name')

    @get_global_defn_name.on(IR.FunctionDefinition)
    def get_global_defn_name(self):
        return f'{self.module_name}#{self.short_name}'

    @get_global_defn_name.on(IR.ClassDefinition)
    def get_global_defn_name(self):
        return f'{self.module_name}#{self.short_name}'

    @get_global_defn_name.on(IR.StaticMethodDefinition)
    def get_global_defn_name(self):
        return f'{get_global_defn_name(self.cls)}#{self.name}'

    @get_global_defn_name.on(IR.InstanceMethodDefinition)
    def get_global_defn_name(self):
        return f'{get_global_defn_name(self.cls)}#{self.name}'

    @D.on(IR.FunctionDefinition)
    def D(self, ctx):
        if self.body is not None:
            emit_function_body(
                ctx=ctx,
                debug_name=self.short_name,
                proto=proto_for(self),
                extern=self.extern,
                rtype=self.rtype,
                start_callback=lambda ctx: copy_params_to_locals(
                    ctx=ctx,
                    paramtypes=[p.type for p in self.plist.params],
                    c_param_names=encode_param_names(self.plist),
                    c_local_names=list(map(cvarname, self.plist.params)),
                ),
                body=self.body
            )

    def emit_global_variable_initializer(self, ctx):
        assert isinstance(self, IR.GlobalVariableDefinition), self
        initializer_proto = get_global_var_initializer_proto(self)
        emit_function_body(
            ctx=ctx,
            debug_name=self.short_name,
            proto='static ' + initializer_proto,
            extern=False,
            rtype=self.type,
            start_callback=lambda ctx: None,
            body=IR.Block(self.token, [], [self.expr]),
        )

    def emit_global_variable_getter(self, ctx):
        assert isinstance(self, IR.GlobalVariableDefinition), self
        getter_proto = get_global_var_getter_proto(self)
        initializer_name = get_global_var_initializer_name(self)
        initvar_name = get_global_initvar_name(self)
        var_name = get_global_var_name(self)

        ctx.out += getter_proto
        ctx.out += '{'
        with ctx.push_indent(1):
            ctx.out += f'if (!{initvar_name})'
            ctx.out += '{'
            with ctx.push_indent(1):
                ctx.out += (
                    f'{declare(ERROR_POINTER_TYPE, ERROR_POINTER_NAME)} = '
                    f'{initializer_name}({STACK_POINTER_NAME}, &{var_name});'
                )
                ctx.out += f'if ({ERROR_POINTER_NAME})'
                ctx.out += '{'
                with ctx.push_indent(1):
                    ctx.out += f'return {ERROR_POINTER_NAME};'
                ctx.out += '}'
                ctx.out += _release_on_exit(self.type, var_name)
                ctx.out += f'{initvar_name} = 1;'
            ctx.out += '}'
            ctx.out += _retain(self.type, var_name)
            ctx.out += f'*{OUTPUT_PTR_NAME} = {var_name};'
            ctx.out += 'return NULL;'
        ctx.out += '}'

    @D.on(IR.GlobalVariableDefinition)
    def D(self, ctx):
        initvar_name = get_global_initvar_name(self)
        var_name = get_global_var_name(self)
        ctx.static_vars += f'static KLC_bool {initvar_name};'
        ctx.static_vars += f'static {declare(self.type, var_name)};'
        emit_global_variable_initializer(self, ctx)
        emit_global_variable_getter(self, ctx)

    @D.on(IR.StructDefinition)
    def D(self, ctx):
        pass

    def emit_malloc_and_deleter(self, ctx):
        assert isinstance(self, IR.ClassDefinition), self
        descriptor_name = get_class_descriptor_name(self)
        struct_name = get_class_struct_name(self)
        malloc_name = get_class_malloc_name(self)
        malloc_type = get_class_malloc_type(self)
        delete_hook_name = get_class_delete_hook_name(self)
        delete_hook_type = get_delete_hook_type(self)
        deleter_name = get_class_deleter_name(self)

        ctx.static_fdecls += f'static {declare(malloc_type, malloc_name)};'

        ctx.out += 'static ' + declare(malloc_type, malloc_name)
        ctx.out += '{'
        with ctx.push_indent(1):
            ctx.out += f'{struct_name} zero = ' '{0};'
            ctx.out += f'{struct_name}* ret = malloc(sizeof({struct_name}));'
            ctx.out += f'*ret = zero;'
            ctx.out += f'ret->header.cls = &{descriptor_name};'
            ctx.out += f'return ret;'
        ctx.out += '}'

        ctx.out += 'static ' + declare(delete_hook_type, delete_hook_name, [
            THIS_NAME, DELETE_QUEUE_NAME,
        ])
        with ctx.retain_scope():
            ctx.declare(ERROR_POINTER_TYPE, ERROR_POINTER_NAME)
            E(self.delete_hook.body, ctx)

        ctx.out += 'static ' + declare(
            DELETER_TYPE, deleter_name, ['robj', 'dq'])
        ctx.out += '{'
        retainable_fields = [
            field for field in self.fields if
            isinstance(field.type, IR.Retainable)
        ]
        with ctx.push_indent(1):
            ctx.out += f'{struct_name}* obj = ({struct_name}*) robj;'
            ctx.out += f'{delete_hook_name}(obj, dq);'
            for field in retainable_fields:
                if isinstance(field.type, IR.ClassDefinition):
                    c_field_name = get_class_c_struct_field_name(field)
                    ctx.out += (
                        f'KLC_partial_release('
                        f'(KLC_Header*) obj->{c_field_name}, dq);'
                    )
                elif field.type == IR.VAR_TYPE:
                    c_field_name = get_class_c_struct_field_name(field)
                    ctx.out += (
                        f'KLC_partial_release_var(obj->{c_field_name}, dq);'
                    )
                else:
                    assert False, field
        ctx.out += '}'

    def emit_methods(self, ctx):
        assert isinstance(self, IR.TraitOrClassDefinition), self

        for static_method in self.static_methods:
            D(static_method, ctx)

        for instance_method in self.instance_methods:
            D(instance_method, ctx)

    def emit_trait_or_class_descriptor(self, ctx):
        assert isinstance(self, IR.TraitOrClassDefinition), self
        method_list_name = get_method_list_name(self)
        descriptor_name = get_class_descriptor_name(self)
        sorted_instance_methods = sorted(
            self.instance_method_closure,
            key=lambda method: method.name,
        )

        if sorted_instance_methods:
            ctx.out += f'KLC_MethodEntry {method_list_name}[] = ' '{'
            with ctx.push_indent(1):
                for instance_method in sorted_instance_methods:
                    ctx.out += '{'
                    with ctx.push_indent(1):
                        method_c_name = (
                            get_instance_method_name(instance_method)
                        )
                        ctx.out += f'"{instance_method.name}",'
                        ctx.out += f'&{method_c_name},'
                    ctx.out += '},'
            ctx.out += '};'

        ctx.out += f'KLC_Class {descriptor_name} = ' '{'
        with ctx.push_indent(1):
            ctx.out += f'"{self.module_name}",'
            ctx.out += f'"{self.short_name}",'
            if isinstance(self, IR.TraitDefinition):
                # Traits are not concrete, and so have no deleter
                ctx.out += f'NULL,'
            else:
                ctx.out += f'&{get_class_deleter_name(self)},'
            if sorted_instance_methods:
                ctx.out += (
                    f'sizeof({method_list_name})/sizeof(KLC_MethodEntry),'
                )
                ctx.out += f'{method_list_name},'
            else:
                ctx.out += '0,'
                ctx.out += 'NULL,'
        ctx.out += '};'

    @D.on(IR.ClassDefinition)
    def D(self, ctx):
        emit_malloc_and_deleter(self, ctx)
        emit_methods(self, ctx)
        emit_trait_or_class_descriptor(self, ctx)

    @D.on(IR.TraitDefinition)
    def D(self, ctx):
        emit_methods(self, ctx)
        emit_trait_or_class_descriptor(self, ctx)

    @D.on(IR.StaticMethodDefinition)
    def D(self, ctx):
        emit_function_body(
            ctx=ctx,
            debug_name=f'{self.cls.short_name}.{self.name}',
            proto=proto_for(self),
            extern=False,
            rtype=self.rtype,
            start_callback=lambda ctx: copy_params_to_locals(
                ctx=ctx,
                paramtypes=[p.type for p in self.plist.params],
                c_param_names=encode_param_names(self.plist),
                c_local_names=list(map(cvarname, self.plist.params)),
            ),
            body=self.body,
        )

    @D.on(IR.InstanceMethodDefinition)
    def D(self, ctx):
        token = self.token
        this_type = self.this_type
        emit_function_body(
            ctx=ctx,
            debug_name=f'{self.cls.short_name}.{self.name}',
            proto=proto_for(self),
            extern=False,
            rtype=IR.VAR_TYPE,
            start_callback=lambda ctx: copy_method_params_to_locals(
                ctx=ctx,
                token=self.token,
                this_type=this_type,
                paramtypes=[p.type for p in self.plist.params],
                c_local_names=list(map(cvarname, self.plist.params)),
            ),
            body=IR.Cast(
                token,
                self.body if self.rtype == IR.VOID else
                    IR.Cast(token, self.body, self.rtype),
                IR.VAR_TYPE,
            ),
        )

    def copy_method_params_to_locals(
            ctx, token, this_type, paramtypes, c_local_names):
        assert len(paramtypes) == len(c_local_names), c_local_names
        argc = len(c_local_names)
        ctx.out += f'if ({METHOD_PARAM_ARGC_NAME} != 1 + {argc})' '{'
        with ctx.push_indent(1):
            ctx.set_error_pointer(f'Method expected 1 + {argc} args')
            ctx.jump_out_of_scope()
        ctx.out += '}'
        ctx.declare(this_type, THIS_NAME)
        converted_this_name = cast_c_name(
            ctx=ctx,
            token=token,
            st=IR.VAR_TYPE,
            dt=this_type,
            c_name=f'{METHOD_PARAM_ARGV_NAME}[0]',
        )
        ctx.out += f'{THIS_NAME} = {converted_this_name};'
        ctx.retain(THIS_NAME)
        for i, (t, c_name) in enumerate(zip(paramtypes, c_local_names), 1):
            ctx.declare(t, c_name)
            converted_name = cast_c_name(
                ctx=ctx,
                token=token,
                st=IR.VAR_TYPE,
                dt=t,
                c_name=f'{METHOD_PARAM_ARGV_NAME}[{i}]',
            )
            ctx.out += f'{c_name} = {converted_name};'
            ctx.retain(c_name)

    proto_for = Multimethod('proto_for')

    @proto_for.on(IR.FunctionDefinition)
    def proto_for(self):
        return declare(
            self.type,
            cvarname(self),
            encode_param_names(self.plist),
        )

    @proto_for.on(IR.StaticMethodDefinition)
    def proto_for(self):
        return declare(
            self.type,
            cvarname(self),
            encode_param_names(self.plist),
        )

    @proto_for.on(IR.InstanceMethodDefinition)
    def proto_for(self):
        return declare(
            METHOD_TYPE,
            cvarname(self),
            [METHOD_PARAM_ARGC_NAME, METHOD_PARAM_ARGV_NAME],
        )

    # translate expressions
    E = Multimethod('E')

    """
    Memory management rules with expressions:

        * All variables declared with 'declare' will be
            auto-released at the end of the given retain_scope

        * Every 'E' definition should balance all the refs
            within its function boundary.
    """

    @E.on(IR.Block)
    def E(self, ctx):
        retvar = (
            ctx.declare(self.type) if self.type != IR.VOID else
            None
        )
        with ctx.retain_scope():
            for decl in self.decls:
                ctx.declare(decl.type, cvarname(decl))

            for expr in self.exprs[:-1]:
                with ctx.retain_scope():
                    E(expr, ctx)
            if self.exprs:
                last_retvar = E(self.exprs[-1], ctx)

            if self.type != IR.VOID:
                assert last_retvar is not None, self.type
                ctx.out += f'{retvar} = {last_retvar};'
                ctx.retain(retvar)

        return retvar

    @E.on(IR.This)
    def E(self, ctx):
        retvar = ctx.declare(self.type)
        ctx.out += f'{retvar} = {THIS_NAME};'
        ctx.retain(retvar)
        return retvar

    @E.on(IR.DeleteQueueName)
    def E(self, ctx):
        assert self.type == IR.VOIDP
        retvar = ctx.declare(IR.VOIDP)
        ctx.out == f'{retvar} = {DELETE_QUEUE_NAME};'
        return retvar

    @E.on(IR.SetLocalName)
    def E(self, ctx):
        cname = cvarname(self.decl)
        retvar = E(self.expr, ctx)
        ctx.retain(retvar)
        ctx.release(cname)
        ctx.out += f'{cname} = {retvar};'
        return cname

    @E.on(IR.LocalName)
    def E(self, ctx):
        return cvarname(self.decl)

    @E.on(IR.GlobalName)
    def E(self, ctx):
        return emit_function_call(
            ctx=ctx,
            token=self.token,
            rtype=self.type,
            c_args=[],
            extern=False,
            c_function_name=get_global_var_getter_name(self.defn),
        )

    @E.on(IR.PointerAndIntArithmetic)
    def E(self, ctx):
        ptr_var = E(self.left, ctx)
        int_var = E(self.right, ctx)
        retvar = ctx.declare(self.type)
        ctx.out += f'{retvar} = {ptr_var} {self.op} {int_var};'
        return retvar

    @E.on(IR.Cast)
    def E(self, ctx):
        if self.type == IR.VOID:
            E(self.expr, ctx)
        else:
            return cast_expr(ctx, self.expr, self.type)

    def cast_expr(ctx, expr, dt):
        c_name = E(expr, ctx)
        return cast_c_name(
            ctx, expr.token, expr.type, dt, c_name)

    def cast_c_name(ctx, token, st, dt, c_name):
        if st == dt:
            return c_name
        else:
            return _cast(st, dt, c_name, ctx, token)

    _cast = Multimethod('_cast', 2)

    def _cast_error(st, dt, token):
        return Error([token], f'Unsupported cast from {st} to {dt}')

    @_cast.on(IR.PrimitiveTypeDefinition, IR.PrimitiveTypeDefinition)
    def _cast(st, dt, c_name, ctx, token):
        assert st != IR.VOID and dt != IR.VOID, (st, dt)
        retvar = ctx.declare(dt)
        if dt == IR.BOOL:
            ctx.out += f'{retvar} = !!({c_name});'
        else:
            cdecl = declare(dt, '')
            ctx.out += f'{retvar} = (({cdecl}) {c_name});'
        return retvar

    @_cast.on(IR.PrimitiveTypeDefinition, type(IR.VAR_TYPE))
    def _cast(st, dt, c_name, ctx, token):
        if st == IR.VOID:
            return ctx.declare(IR.VAR_TYPE)

        if st in IR.INTEGRAL_TYPES:
            convert_func = 'KLC_var_from_int'
        elif st in IR.FLOAT_TYPES:
            convert_func = 'KLC_var_from_float'
        elif st == IR.TYPE:
            convert_func = f'KLC_var_from_type'
        else:
            assert False, st

        retvar = ctx.declare(IR.VAR_TYPE)
        ctx.out += f'{retvar} = {convert_func}({c_name});'
        return retvar

    @_cast.on(type(IR.VAR_TYPE), IR.ClassDefinition)
    def _cast(st, dt, c_name, ctx, token):
        descriptor_name = get_class_descriptor_name(dt)
        tvar = ctx.declare(HEADER_POINTER_TYPE)
        retvar = ctx.declare(dt)
        with stack_trace_entry(ctx, token):
            ctx.out += (
                f'{ERROR_POINTER_NAME} = '
                f'KLC_var_to_ptr({STACK_POINTER_NAME}, '
                f'&{tvar}, {c_name}, &{descriptor_name});'
            )
        cdecl = declare(dt, '')
        ctx.out += f'{retvar} = (({cdecl}) {tvar});'
        ctx.retain(retvar)
        ctx.jump_on_error()
        return retvar

    @_cast.on(type(IR.VAR_TYPE), IR.PrimitiveTypeDefinition)
    def _cast(st, dt, c_name, ctx, token):

        if dt == IR.VOID:
            return

        if dt in IR.INTEGRAL_TYPES:
            convert_func = 'KLC_var_to_int'
            extract_type = IR.INT
        elif dt in IR.FLOAT_TYPES:
            convert_func = 'KLC_var_to_float'
            extract_type = IR.FLOAT
        elif dt == IR.TYPE:
            convert_func = 'KLC_var_to_type'
            extract_type = IR.TYPE
        else:
            assert False, dt

        vvar = c_name
        tvar = ctx.declare(extract_type)

        if extract_type != dt:
            retvar = ctx.declare(dt)
        else:
            retvar = tvar

        with stack_trace_entry(ctx, token):
            ctx.out += (
                f'{ERROR_POINTER_NAME} = '
                f'{convert_func}({STACK_POINTER_NAME}, '
                f'&{tvar}, {vvar});'
            )
        ctx.jump_on_error()
        cdecl = declare(dt, '')

        if extract_type != dt:
            ctx.out += f'{retvar} = (({cdecl}) {tvar});'

        return retvar

    @_cast.on(IR.ClassDefinition, type(IR.VAR_TYPE))
    def _cast(st, dt, c_name, ctx, token):
        retvar = ctx.declare(IR.VAR_TYPE)
        ctx.out += f'{retvar} = KLC_var_from_ptr((KLC_Header*) {c_name});'
        ctx.retain(retvar);
        return retvar

    @_cast.on(IR.PointerType, IR.PointerType)
    def _cast(st, dt, c_name, ctx, token):
        retvar = ctx.declare(dt)
        ctx.out += f'{retvar} = (({declare(dt, "")}) {c_name});'
        ctx.retain(retvar)
        return retvar

    @_cast.on(IR.Type, IR.Type)
    def _cast(st, dt, c_name, ctx, token):
        raise _cast_error(st, dt, token)

    def _get_struct_field_chain(expr):
        reverse_field_chain = []
        while isinstance(expr, IR.GetStructField):
            reverse_field_chain.append(expr.field_defn)
            expr = expr.expr
        assert isinstance(expr.type, IR.StructDefinition), expr
        return expr, list(reversed(reverse_field_chain))

    @E.on(IR.GetStructField)
    def E(self, ctx):
        assert not isinstance(self.type, IR.Retainable), self
        struct_expr, field_defns = _get_struct_field_chain(self)
        c_field_names = [
            get_c_struct_field_name(defn) for defn in field_defns
        ]
        c_field_chain = '.'.join(c_field_names)
        structvar = E(struct_expr, ctx)
        retvar = ctx.declare(self.type)
        cfname = get_c_struct_field_name(self.field_defn)
        ctx.out += f'{retvar} = {structvar}.{c_field_chain};'
        return retvar

    @E.on(IR.SetStructField)
    def E(self, ctx):
        assert not isinstance(self.type, IR.Retainable), self
        struct_expr, field_defns = _get_struct_field_chain(self.expr)
        c_field_names = [
            get_c_struct_field_name(defn) for defn in field_defns
        ] + [
            get_c_struct_field_name(self.field_defn),
        ]
        c_field_chain = '.'.join(c_field_names)
        structvar = E(struct_expr, ctx)
        resultvar = E(self.valexpr, ctx)
        retvar = ctx.declare(self.type)
        cfname = get_c_struct_field_name(self.field_defn)
        ctx.out += f'{retvar} = {resultvar};'
        ctx.out += f'{structvar}.{c_field_chain} = {retvar};'
        return retvar

    @E.on(IR.GetClassField)
    def E(self, ctx):
        c_field_name = get_class_c_struct_field_name(self.field_defn)
        this_var = E(self.expr, ctx)
        retvar = ctx.declare(self.type)
        ctx.out += f'{retvar} = {this_var}->{c_field_name};'
        ctx.retain(retvar)
        return retvar

    @E.on(IR.SetClassField)
    def E(self, ctx):
        c_field_name = get_class_c_struct_field_name(self.field_defn)
        this_var = E(self.expr, ctx)
        val_var = E(self.valexpr, ctx)
        retvar = ctx.declare(self.type)
        release_var = ctx.declare(self.field_defn.type)

        field_expr_str = f'{this_var}->{c_field_name}'

        ctx.retain(retvar)
        ctx.out += f'{release_var} = {field_expr_str};'
        ctx.release(release_var, clear_var=True)
        ctx.out += f'{retvar} = {val_var};'
        ctx.out += f'{field_expr_str} = {retvar};'
        ctx.retain(retvar)
        return retvar

    @E.on(IR.TraitOrClassName)
    def E(self, ctx):
        retvar = ctx.declare(self.type)
        descriptor_name = get_class_descriptor_name(self.cls)
        ctx.out += f'{retvar} = &{descriptor_name};'
        return retvar

    @E.on(IR.IntLiteral)
    def E(self, ctx):
        retvar = ctx.declare(IR.INT)
        ctx.out += f'{retvar} = {self.value};'
        return retvar

    @E.on(IR.FloatLiteral)
    def E(self, ctx):
        retvar = ctx.declare(IR.FLOAT)
        ctx.out += f'{retvar} = {self.value};'
        return retvar

    def escape_str(s):
        return (
            s
                .replace('\\', '\\\\')
                .replace('\t', '\\t')
                .replace('\n', '\\n')
                .replace('\r', '\\r')
                .replace('"', '\\"')
                .replace("'", "\\'")
        )

    @E.on(IR.CStringLiteral)
    def E(self, ctx):
        return f'"{escape_str(self.value)}"'

    @contextlib.contextmanager
    def stack_trace_entry(ctx, token):
        ctx.out += (
            f'KLC_stack_push({STACK_POINTER_NAME}, '
            f'{DEBUG_FILE_NAME_NAME}, '
            f'{DEBUG_FUNC_NAME_NAME}, '
            f'{token.lineno});'
        )
        yield
        ctx.out += f'KLC_stack_pop({STACK_POINTER_NAME});'

    def emit_function_call(
            *,
            ctx,
            token,
            rtype,
            c_args,
            extern,
            c_function_name):
        argvars = ', '.join(c_args)

        if extern:
            if rtype == IR.VOID:
                ctx.out += f'{c_function_name}({argvars});'
                return
            else:
                # Function calls implicitly generate a retain,
                # so there's no need to explicitly retain here.
                # And of course, the release is implicit with
                # the 'declare'.
                retvar = ctx.declare(rtype)
                ctx.out += f'{retvar} = {c_function_name}({argvars});'
                return retvar
        else:
            # Function calls implicitly generate a retain,
            # so there's no need to explicitly retain here.
            # And of course, the release is implicit with
            # the 'declare'.
            if rtype == IR.VOID:
                retvar = None
                retvarp = 'NULL'
            else:
                retvar = ctx.declare(rtype)
                retvarp = f'&{retvar}'

            protcol_args = f'{STACK_POINTER_NAME}, {retvarp}'

            margvars = (
                f'{protcol_args}, {argvars}' if c_args else protcol_args
            )

            with stack_trace_entry(ctx, token):
                ctx.out += (
                    f'{ERROR_POINTER_NAME} = {c_function_name}({margvars});'
                )
            ctx.jump_on_error()
            return retvar

    @E.on(IR.FunctionCall)
    def E(self, ctx):
        fvar = E(self.f, ctx)
        return emit_function_call(
            ctx=ctx,
            token=self.token,
            rtype=self.type,
            c_args=[E(arg, ctx) for arg in self.args],
            extern=self.f.type.extern,
            c_function_name=fvar,
        )

    @E.on(IR.StaticMethodCall)
    def E(self, ctx):
        return emit_function_call(
            ctx=ctx,
            token=self.token,
            rtype=self.type,
            c_args=[E(arg, ctx) for arg in self.args],
            extern=False,
            c_function_name=get_static_method_name(self.f),
        )

    @E.on(IR.InstanceMethodCall)
    def E(self, ctx):
        # Including the 'this' argument, self.args should never be
        # empty.
        assert self.args, self.args
        c_arg_names = [E(arg, ctx) for arg in self.args]

        arg_array_name = ctx._new_unique_cname()
        ctx._decls += f'KLC_var {arg_array_name}[{len(self.args)}];'

        for i, c_arg_name in enumerate(c_arg_names):
            ctx.out += f'{arg_array_name}[{i}] = {c_arg_name};'

        return emit_function_call(
            ctx=ctx,
            token=self.token,
            rtype=IR.VAR_TYPE,
            c_args=[
                f'"{escape_str(self.name)}"',
                str(len(self.args)),
                arg_array_name,
            ],
            extern=False,
            c_function_name=CALL_METHOD_FUNCTION_NAME,
        )

    @E.on(IR.PrimitiveUnop)
    def E(self, ctx):
        assert isinstance(self.type, IR.PrimitiveTypeDefinition), self.type
        exprvar = E(self.expr, ctx)
        retvar = ctx.declare(self.type)
        ctx.out += f'{retvar} = {self.op} {exprvar};'
        return retvar

    @E.on(IR.PrimitiveBinop)
    def E(self, ctx):
        assert isinstance(self.type, IR.PrimitiveTypeDefinition), self.type
        leftvar = E(self.left, ctx)
        rightvar = E(self.right, ctx)
        retvar = ctx.declare(self.type)
        ctx.out += f'{retvar} = {leftvar} {self.op} {rightvar};'
        return retvar

    @E.on(IR.If)
    def E(self, ctx):
        if self.type != IR.VOID:
            retvar = ctx.declare(self.type)

        with ctx.retain_scope():
            condvar = E(self.cond, ctx)

            ctx.out += f'if ({condvar})'
            with ctx.retain_scope():
                leftvar = E(self.left, ctx)
                if self.type != IR.VOID:
                    ctx.out += f'{retvar} = {leftvar};'
                    ctx.retain(retvar)
            ctx.out += 'else'
            with ctx.retain_scope():
                rightvar = E(self.right, ctx)
                if self.type != IR.VOID:
                    ctx.out += f'{retvar} = {rightvar};'
                    ctx.retain(retvar)

        if self.type != IR.VOID:
            return retvar

    @E.on(IR.While)
    def E(self, ctx):
        assert self.cond.type == IR.BOOL, self.cond.type

        condvar = ctx.declare(IR.BOOL)
        ctx.out += f'{condvar} = 1;'
        ctx.out += f'while ({condvar})'
        with ctx.retain_scope():
            saved_condvar = ctx.declare(self.cond.type)

            with ctx.retain_scope():
                computed_condvar = E(self.cond, ctx)
                ctx.out += f'{condvar} = {computed_condvar};'

            ctx.out += f'if (!{condvar})'
            ctx.out += '{'
            with ctx.push_indent(1):
                ctx.jump_out_of_scope()
            ctx.out += '}'

            E(self.body, ctx)

    @E.on(IR.LogicalAnd)
    def E(self, ctx):
        retvar = ctx.declare(IR.BOOL)

        with ctx.retain_scope():
            leftvar = E(self.left, ctx)
            ctx.out += f'if ({leftvar})'
            with ctx.retain_scope():
                rightvar = E(self.right, ctx)
                ctx.out += f'{retvar} = {rightvar};'
            ctx.out += 'else'
            with ctx.retain_scope():
                ctx.out += f'{retvar} = {leftvar};'

        return retvar

    @E.on(IR.LogicalOr)
    def E(self, ctx):
        retvar = ctx.declare(IR.BOOL)

        with ctx.retain_scope():
            leftvar = E(self.left, ctx)
            ctx.out += f'if ({leftvar})'
            with ctx.retain_scope():
                ctx.out += f'{retvar} = {leftvar};'
            ctx.out += 'else'
            with ctx.retain_scope():
                rightvar = E(self.right, ctx)
                ctx.out += f'{retvar} = {rightvar};'

        return retvar

    @E.on(IR.Malloc)
    def E(self, ctx):
        malloc_name = get_class_malloc_name(self.type)
        retvar = ctx.declare(self.type)
        ctx.out += f'{retvar} = {malloc_name}();'
        return retvar

    @E.on(IR.ThrowStringLiteral)
    def E(self, ctx):
        ctx.set_error_pointer(self.value)
        ctx.jump_out_of_scope()

    @E.on(IR.FunctionName)
    def E(self, ctx):
        return cvarname(self.defn)

    # For initializing variables when declaring them
    init_expr_for = Multimethod('init_expr_for')

    @init_expr_for.on(IR.StructDefinition)
    def init_expr_for(self):
        return '{0}'

    @init_expr_for.on(IR.ClassDefinition)
    def init_expr_for(self):
        return 'NULL'

    @init_expr_for.on(type(IR.VAR_TYPE))
    def init_expr_for(self):
        return '{0, {0}}'

    @init_expr_for.on(IR.PointerType)
    def init_expr_for(self):
        return 'NULL'

    @init_expr_for.on(IR.PrimitiveTypeDefinition)
    def init_expr_for(self):
        return '0'


def main():
    aparser = argparse.ArgumentParser()
    aparser.add_argument('filename')
    aparser.add_argument('--search-dir', default='srcs')
    aparser.add_argument('--out-dir', default='out')
    aparser.add_argument('--operation', '-p', default='translate', choices=(
        'parse',
        'translate',
    ))
    args = aparser.parse_args()
    source = Source.from_name_and_path('main', args.filename)
    module_table = parser.parse(source, search_dir=args.search_dir)

    if args.operation == 'parse':
        for module in module_table.values():
            print(module.format())

    tu_table = {
        name: C.translate(module) for name, module in module_table.items()
    }

    for tu in tu_table.values():
        C.write_out(tu, out_dir=args.out_dir)

    if args.operation == 'translate':
        return


if __name__ == '__main__':
    main()

