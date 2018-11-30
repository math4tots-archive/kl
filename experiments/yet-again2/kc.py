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
      'is', 'not', 'null', 'true', 'false', 'new', 'delete',
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
      '||', '&&', '|', '&', '<<', '>>', '~', '...', '$',
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
        def __init__(self, token, *args):
            self.token = token
            for (fname, ftype), arg in zip(type(self).node_fields, args):
                if not isinstance(arg, ftype):
                    raise TypeError(
                        'Expected type of %r to be %r, but got %r' % (
                            fname, ftype, arg))
                setattr(self, fname, arg)
            if len(type(self).node_fields) != len(args):
                raise TypeError('%s expects %s arguments, but got %s' % (
                    type(self).__name__,
                    len(type(self).node_fields),
                    len(args),
                ))

        def __repr__(self):
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
                Node._map_helper(getattr(self, fieldname))
                for fieldname, _ in nt.node_fields
            ])

        @classmethod
        def _map_helper(cls, f, value):
            if isinstance(value, (type(None), int, float, bool, str, IR.CType)):
                return value
            if isinstance(value, list):
                return [Node._map_helper(x) for x in value]
            if isinstance(value, tuple):
                return tuple(Node._map_helper(x) for x in value)
            if isinstance(value, set):
                return {Node._map_helper(x) for x in value}
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
    class Type:
        """Marker class for what values indicate Types.
        """

        # Only 'complete' types can be directly declared.
        # This is mostly to prevent structs from being declared
        # before their definition is processed.
        complete = True

        @property
        def is_meta_type(self):
            return isinstance(self, PseudoType)

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
        pass

    @ns
    class FunctionLike:
        """
        abstract
            extern: bool
            rtype: type
            params: typeutil.List[Parameter]
            vararg: bool
            body: typeutil.Optional[Block]
        """
        @lazy(lambda: FunctionType)
        def type(self):
            return FunctionType(
                self.extern,
                self.rtype,
                [p.type for p in self.params],
                self.vararg,
            )

    @ns
    class FunctionDefinition(GlobalDefinition, Declaration, FunctionLike):
        node_fields = (
            ('extern', bool),
            ('rtype', Type),
            ('module_name', str),
            ('short_name', str),
            ('params', typeutil.List[Parameter]),
            ('vararg', bool),
            ('has_body', bool),
            ('body_promise', typeutil.Optional[Promise]),
        )

        @lazy(lambda: typeutil.Optional[Block])
        def body(self):
            return (
                self.body_promise.resolve()
                    if self.body_promise else None
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
    class StructDefinition(StructOrClassDefinition):
        complete = False

    @ns
    class ClassDefinition(StructOrClassDefinition):
        node_fields = StructOrClassDefinition.node_fields + (
            ('delete_hook_promise', Promise),
            ('static_method_promises', typeutil.List[Promise]),
        )

        @lazy(lambda: DeleteHook)
        def delete_hook(self):
            return self.delete_hook_promise.resolve()

        @lazy(lambda: typeutil.List[StaticMethodDefinition])
        def static_methods(self):
            return [p.resolve() for p in self.static_method_promises]

    @ns
    class StaticMethodDefinition(Declaration, FunctionLike):
        extern = False
        node_fields = (
            ('cls_promise', Promise),
            ('rtype_promise', Promise),
            ('name', str),
            ('params_promise', Promise),
            ('vararg_promise', Promise),
            ('body_promise', Promise),
        )

        @lazy(lambda: Type)
        def rtype(self):
            return self.rtype_promise.resolve()

        @lazy(lambda: ClassDefinition)
        def cls(self):
            return self.cls_promise.resolve()

        @lazy(lambda: typeutil.List[Parameter])
        def params(self):
            return self.params_promise.resolve()

        @lazy(lambda: bool)
        def vararg(self):
            return self.vararg_promise.resolve()

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
    class GlobalVariableDeclaration(VariableDeclaration):
        node_fields = (
            ('extern', bool),
            ('type', Type),
            ('module_name', str),
            ('short_name', str),
        )

    @ns
    class LocalVariableDeclaration(BaseLocalVariableDeclaration):
        mutable = True

    @ns
    class PrimitiveTypeDeclaration(TypeDeclaration):
        node_fields = (
            ('name', str),
        )

    class ProxyMixin:
        def __eq__(self, other):
            return type(self) is type(other) and self._proxy == other._proxy

        def __hash__(self):
            return hash((type(self), self._proxy))

        def __repr__(self):
            return f'{type(self).__name__}{self._proxy}'

    @ns
    class Retainable:
        "Type mixin that indicates the given type is reference counted"

    @ns
    class PointerType(Type, ProxyMixin):
        def __init__(self, base):
            self.base = base

        @property
        def _proxy(self):
            return (self.base,)

    @ns
    class FunctionType(Type, ProxyMixin):
        def __init__(self, extern, rtype, paramtypes, vararg):
            self.extern = extern
            self.rtype = rtype
            self.paramtypes = tuple(paramtypes)
            self.vararg = vararg

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

    PRIMITIVE_TYPE_MAP = {
        t: PrimitiveTypeDeclaration(builtin_token, t)
        for t in lexer.PRIMITIVE_TYPE_NAMES
    }
    ns(PRIMITIVE_TYPE_MAP, 'PRIMITIVE_TYPE_MAP')

    VOID = PRIMITIVE_TYPE_MAP['void']
    ns(VOID, 'VOID')

    VOIDP = PointerType(VOID)
    ns(VOIDP, 'VOIDP')

    convertible = Multimethod('convertible', 2)
    ns(convertible, 'convertible')

    @convertible.on(PrimitiveTypeDeclaration, PrimitiveTypeDeclaration)
    def convertible(a, b):
        return a == b or (
            (a.name, b.name) in {
                ('int', 'long'),
                ('int', 'size_t'),
                ('long', 'size_t'),
            }
        ) or (
            frozenset((a.name, b.name)) in set(map(frozenset, [
                ['char', 'unsigned char'],
                ['char', 'signed char'],
                ['unsigned char', 'signed char'],
            ]))
        )

    @convertible.on(Type, Type)
    def convertible(a, b):
        return a == b

    @ns
    class PrimitiveTypeDefinition(GlobalDefinition):
        node_fields = ()

    @ns
    class Expression(Node):
        """
        abstract
            type: Type
        """

        @property
        def is_pseudo_expression(self):
            return isinstance(self, PseudoExpression)

    @ns
    class PseudoExpression(Expression):
        """Expressions that are not themselves concrete values.
        """

        @property
        def type(self):
            return PSEUDO_TYPE

    @ns
    class TypeExpression(PseudoExpression):
        pass

    @ns
    class ClassName(TypeExpression):
        node_fields = (
            ('cls', ClassDefinition),
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
            ('exprs', typeutil.List[Expression]),
        )

        @property
        def type(self):
            return self.exprs[-1].type if self.exprs else VOID

    @ns
    class FunctionName(Expression):
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
            ('expr', Expression),
        )

        @property
        def type(self):
            return self.decl.type

    @ns
    class FieldDefinition(Node):
        node_fields = (
            ('extern', bool),
            ('type', Type),
            ('name', str),
        )

    @ns
    class GetStructField(Expression):
        node_fields = (
            ('field_defn', FieldDefinition),
            ('expr', Expression),
        )

        @property
        def type(self):
            return self.field_defn.type

    @ns
    class SetStructField(Expression):
        node_fields = (
            ('field_defn', FieldDefinition),
            ('expr', Expression),
            ('valexpr', Expression),
        )

        @property
        def type(self):
            return self.field_defn.type

    @ns
    class GetClassField(Expression):
        node_fields = (
            ('field_defn', FieldDefinition),
            ('expr', Expression),
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
            ('from_extern', bool),  # iff we are calling form extern func
            ('to_extern', bool),    # iff are are calling an extern function
            ('type', Type),
            ('f', Expression),
            ('args', typeutil.List[Expression]),
        )

    @ns
    class StaticMethodCall(Expression):
        node_fields = (
            ('from_extern', bool),
            ('f', StaticMethodDefinition),
            ('args', typeutil.List[Expression]),
        )

        @property
        def type(self):
            return self.f.rtype

    @ns
    class Malloc(Expression):
        node_fields = (
            ('type', Type),
        )

    @ns
    class This(Expression):
        node_fields = (
            ('cls', ClassDefinition),
        )

        @property
        def type(self):
            return self.cls

        def __repr__(self):
            return f'This({self.cls.module_name}.{self.cls.short_name})'

    @ns
    class IntLiteral(Expression):
        type = PRIMITIVE_TYPE_MAP['int']

        node_fields = (
            ('value', int),
        )

    @ns
    class DoubleLiteral(Expression):
        type = PRIMITIVE_TYPE_MAP['double']

        node_fields = (
            ('value', float),
        )

    @ns
    class StringLiteral(Expression):
        type = PointerType(ConstType(PRIMITIVE_TYPE_MAP['char']))

        node_fields = (
            ('value', str),
        )

    @ns
    class PointerCast(Expression):
        node_fields = (
            ('type', PointerType),
            ('expr', Expression),
        )

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
    class GlobalVariableDefinition(GlobalDefinition):
        node_fields = (
            ('decl', GlobalVariableDeclaration),
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
        def __init__(self, token, extern, this_type):
            super().__init__(token, extern, this_type)

        node_fields = (
            # Indicates whether we're in an 'extern' context.
            # E.g. we cannot throw exceptions from an extern
            # context. Also the calling convention is different.
            ('extern', bool),

            ('this_type', typeutil.Optional[ClassDefinition])
        )

    def new_builtin_extern_struct(name):
        defn = StructDefinition(
            builtin_token,
            True,
            'builtin',
            'KLC_Header',
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
            if not IR.convertible(expr.type, type):
                with self.push(expr.token):
                    raise self.error(
                        f'{expr.type} is not convertible to {type}')
            return expr

        def from_import(self, token, module_name, exported_name, alias):
            with self.push(token):
                exported_defn = (
                    self.load_scope_for(module_name)[exported_name]
                )
            self[alias] = exported_defn
            return Promise.value(
                IR.FromImport(token, module_name, exported_name, alias),
            )

    BUILTINS_MODULE_NAME = 'builtins'

    _builtins_export_names = (
        'String',
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
                    raise module_scope.error(f'Expected {t} but got {peek()}')
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
                with scope.push(token):
                    type = scope[name]
                if not isinstance(type, IR.Type):
                    with scope.push(token), scope.push(type):
                        raise scope.error(f'{name} is not a type')
                return type
            return promise

        def parse_type(scope):
            token = peek()
            if token.type in lexer.C_PRIMITIVE_TYPE_SPECIFIERS:
                parts = [gettok().type]
                while peek().type in lexer.C_PRIMITIVE_TYPE_SPECIFIERS:
                    parts.append(gettok().type)
                name = ' '.join(parts)
                if name not in IR.PRIMITIVE_TYPE_MAP:
                    with scope.push(token):
                        raise scope.error(f'{repr(name)} is not a type')
                type_promise = (
                    Promise.value(IR.PRIMITIVE_TYPE_MAP[name])
                )
            else:
                name = expect_id()
                type_promise = promise_type_from_name(scope, token, name)
            while True:
                token = peek()
                if consume('*'):
                    type_promise = pcall(IR.PointerType, type_promise)
                elif consume('const'):
                    type_promise = pcall(IR.ConstType, type_promise)
                else:
                    break
            return type_promise

        def parse_params(scope):
            expect('(')
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
            return Promise(lambda: ([p.resolve() for p in paramps], vararg))

        def at_variable_declaration():
            if peek().type in lexer.C_PRIMITIVE_TYPE_SPECIFIERS:
                return True

            # Whitelist a few patterns as being the start
            # of a variable declaration
            seqs = [
                ['NAME', 'NAME', '='],
                ['NAME', 'NAME', '\n'],
                ['NAME', '*', '*'],
                ['NAME', '*', 'NAME', '='],
                ['NAME', '*', 'NAME', '\n'],
                ['NAME', '!'],
                ['NAME', 'const'],
            ]
            for seq in seqs:
                if [t.type for t in tokens[i:i+len(seq)]] == seq:
                    return True

            return False

        def promise_set_local_name(scope, token, decl_promise, expr_promise):
            @Promise
            def promise():
                decl = decl_promise.resolve()
                expr = expr_promise.resolve()
                assert isinstance(decl, IR.BaseLocalVariableDeclaration), decl
                if not decl.mutable:
                    with scope.push(token), scope.push(decl.token):
                        raise scope.error(
                            f'Tried to assign to non-mutable variable')
                return IR.SetLocalName(token, decl, expr)
            return promise

        def check_concrete_expression(scope, expr):
            if expr.type.is_meta_type or expr.is_pseudo_expression:
                with scope.push(expr.token):
                    raise scope.error(f'{expr} is not a value')
            return expr

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
                    consume_all('\n')
            @Promise
            def promise():
                _check_has_expression_context(parent_scope)
                return IR.Block(
                    token,
                    [p.resolve() for p in decl_promises],
                    [
                        check_concrete_expression(scope, p.resolve())
                        for p in expr_promises
                    ],
                )
            return promise

        def _check_has_expression_context(scope):
            assert isinstance(scope['@ec'], IR.ExpressionContext)

        def parse_expression(scope):
            promise = parse_postfix(scope)
            @promise.map
            def promise(expr):
                _check_has_expression_context(scope)
                return expr
            return promise

        def parse_args(scope):
            argps = []
            expect('(')
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
                        scope['@ec'].extern,
                        f.type.extern,
                        f.type.rtype,
                        f,
                        args,
                    )
                elif isinstance(f, IR.StaticMethodName):
                    args = IR.convert_args(f.defn.type, scope, token, raw_args)
                    return IR.StaticMethodCall(
                        token,
                        scope['@ec'].extern,
                        f.defn,
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
                if isinstance(expr, IR.ClassName):
                    defn = get_static_method(scope, token, expr.cls, fname)
                    return IR.StaticMethodName(token, defn)
                elif isinstance(expr.type, IR.StructDefinition):
                    defn = get_field_defn(scope, token, expr.type, fname)
                    return IR.GetStructField(
                        token,
                        defn,
                        expr,
                    )
                elif isinstance(expr.type, IR.ClassDefinition):
                    defn = get_field_defn(scope, token, expr.type, fname)
                    return IR.GetClassField(
                        token,
                        defn,
                        expr,
                    )
                else:
                    with scope.push(token):
                        raise scope.error(
                            f'{expr.type} is not a struct or class type'
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
                        defn,
                        expr,
                        val,
                    )
                elif isinstance(expr.type, IR.ClassDefinition):
                    return IR.SetClassField(
                        token,
                        defn,
                        expr,
                        val,
                    )
                else:
                    with scope.push(token):
                        raise scope.error(
                            f'{expr.type} is not a struct or class type'
                        )
            return promise

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
                    if consume('='):
                        valp = parse_expression(scope)
                        expr = promise_set_field(
                            scope, token, expr, name, valp)
                    else:
                        expr = promise_get_field(scope, token, expr, name)
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
                if isinstance(defn, IR.ClassDefinition):
                    return IR.ClassName(token, defn)
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
            if isinstance(type, IR.PointerType):
                if isinstance(arg.type, IR.PointerType):
                    return IR.PointerCast(
                        token,
                        type,
                        arg,
                    )
                else:
                    with scope.push(token):
                        raise scope.error(
                            f'Only pointer can be cast to other pointer '
                            f'types but got {arg.type}')
            with scope.push(token):
                raise scope.error(
                    f'Only pointer casts are supported right now')

        def parse_primary(scope):
            token = peek()
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
                    return expr
            if consume('this'):
                return promise_this(scope, token)
            if consume('NAME'):
                name = token.value
                return promise_name(scope, token, name)
            if consume('INT'):
                return Promise.value(IR.IntLiteral(token, token.value))
            if consume('FLOAT'):
                return Promise.value(IR.DoubleLiteral(token, token.value))
            if consume('STRING'):
                return Promise.value(IR.StringLiteral(token, token.value))
            if consume('throw'):
                # For now, only allow string literals
                value = expect('STRING').value
                return promise_throw_string_literal(scope, token, value)
            if consume('static_cast'):
                expect('(')
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
                if not field.type.complete:
                    with scope.push(field.token):
                        raise scope.error(
                            f'{field.type} is an incomplete type'
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
                defn.complete = True
                return defn

            return promise

        def parse_class(outer_scope, token, extern):
            assert not extern, 'extern is not meaningful for a class'
            name = expect_id()

            field_promises = []
            static_method_promises = []

            defn = IR.ClassDefinition(
                token,
                extern,
                module_name,
                name,
                field_promises,
                Promise(lambda: delete_hook_ptr[0]),
                static_method_promises,
            )
            outer_scope[name] = defn

            class_scope = Scope(outer_scope)

            delete_hook_ptr = [None]
            expect('{')
            consume_all('\n')
            while not consume('}'):
                member_token = peek()
                if consume('delete'):
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
                    field_promises.append(pcall(
                        IR.FieldDefinition,
                        member_token,
                        False,
                        member_type_promise,
                        member_name,
                    ))
                    expect('\n')
                consume_all('\n')

            if delete_hook_ptr[0] is None:
                delete_hook_ptr[0] = IR.DeleteHook(
                    token,
                    Promise.value(IR.Block(token, [], [])),
                )

            @Promise
            def promise():
                fields = defn.fields
                check_member_names(outer_scope, fields)
                check_fields(
                    outer_scope,
                    fields,
                    allow_retainable_fields=True,
                )
                return defn

            return promise

        def parse_static_method(class_scope, token, cls):
            method_scope = Scope(class_scope)
            rtype_promise = parse_type(class_scope)
            name = expect_id()
            params_and_vararg_promise = parse_params(method_scope)
            params_promise = (
                params_and_vararg_promise.map(lambda xs: xs[0])
            )
            vararg_promise = (
                params_and_vararg_promise.map(lambda xs: xs[1])
            )
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
                params_promise,
                vararg_promise,
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
            delete_hook_ptr[0] = IR.DeleteHook(
                token,
                parse_block(delete_scope),
            )
            return delete_hook_ptr[0]

        def parse_function(outer_scope, token, extern, type_promise, name):
            func_scope = Scope(outer_scope)
            params_and_vararg_promise = parse_params(func_scope)
            params_promise = (
                params_and_vararg_promise.map(lambda xs: xs[0])
            )
            vararg_promise = (
                params_and_vararg_promise.map(lambda xs: xs[1])
            )
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
                        body = outer_scope.convert(raw_body, defn.rtype)
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
                params_promise.resolve(),
                vararg_promise.resolve(),
                has_body,
                body_promise,
            ))
            outer_scope.set_promise(token, name, defn_promise)
            return defn_promise

        def parse_global_var_defn(scope, token, extern, type_promise, name):
            expect('\n')
            decl_promise = pcall(
                IR.GlobalVariableDeclaration,
                token,
                extern,
                type_promise,
                module_name,
                name,
            )
            scope.set_promise(token, name, decl_promise)
            return pcall(
                IR.GlobalVariableDefinition,
                token,
                decl_promise,
            )

        def parse_global(scope):
            token = peek()
            extern = bool(consume('extern'))
            if consume('typedef'):
                expect('*')
                name = expect_id()
                scope[name] = IR.PrimitiveTypeDeclaration(token, name)
                return Promise.value(IR.PrimitiveTypeDefinition(token))
            elif consume('struct'):
                return parse_struct(scope, token, extern)
            elif consume('class'):
                return parse_class(scope, token, extern)
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
    ENCODED_GLOBAL_VARIABLE_PREFIX = 'KLCGV'
    ENCODED_LOCAL_VARIABLE_PREFIX = 'KLCLV'
    ENCODED_STRUCT_PREFIX = 'KLCST'
    ENCODED_STRUCT_FIELD_PREFIX = 'KLCSF'
    ENCODED_CLASS_PROTO_NAME = 'KLCCP'
    ENCODED_CLASS_MALLOC_PREFIX = 'KLCCM'
    ENCODED_CLASS_DELETE_HOOK_PREFIX = 'KLCDH'
    ENCODED_CLASS_DELETER_PREFIX = 'KLCCD'
    ENCODED_CLASS_STRUCT_PREFIX = 'KLCCS'
    ENCODED_CLASS_FIELD_PREFIX = 'KLCCF'
    ENCODED_STATIC_METHOD_PREFIX = 'KLCSM'
    OUTPUT_PTR_NAME = 'KLC_output_ptr'
    CLASS_HEADER_FIELD_NAME = 'header'
    HEADER_STRUCT_NAME = 'KLC_Header'
    ERROR_POINTER_NAME = 'KLC_error'
    ERROR_POINTER_TYPE = IR.ERROR_POINTER_TYPE
    HEADER_POINTER_TYPE = IR.HEADER_POINTER_TYPE
    DELETER_TYPE = IR.FunctionType(
        extern=True,
        rtype=IR.VOID,
        paramtypes=[
            HEADER_POINTER_TYPE,
            IR.PointerType(HEADER_POINTER_TYPE),
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

    def _retain(type, cname):
        if isinstance(type, IR.ClassDefinition):
            return f'KLC_retain((KLC_Header*) {cname});'
        else:
            return ''

    def _release(type, cname):
        if isinstance(type, IR.ClassDefinition):
            return f'KLC_release((KLC_Header*) {cname});'
        else:
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
            if isinstance(defn, IR.GlobalVariableDefinition):
                if not defn.decl.extern:
                    gvardecls += f'extern {proto_for(defn)};'
            elif isinstance(defn, IR.FunctionDefinition):
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
            elif isinstance(defn, IR.ClassDefinition):
                assert not defn.extern, defn
                struct_name = get_class_struct_name(defn)
                proto_name = get_class_proto_name(defn)

                fwdstruct += f'typedef struct {struct_name} {struct_name};'

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

                gvardecls += f'extern KLC_Class {proto_name};'

                for static_method in defn.static_methods:
                    fdecls += f'extern {proto_for(static_method)};'

            else:
                raise Fubar([], defn)

        return str(msb)

    def translate_source(module: IR.Module) -> str:
        ctx = Context()
        ctx.out += f'#include "{relative_header_path_from_name(module.name)}"'
        ctx.static_fdecls = ctx.out.spawn()
        for defn in module.definitions:
            D(defn, ctx)
        return str(ctx.out)

    def qualify(module_name, name, prefix):
        return (
            encode(name, prefix=prefix) if module_name is None else
            encode(f'{module_name}.{name}', prefix=prefix)
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

    def get_class_proto_name(decl: IR.Declaration):
        assert isinstance(decl, IR.ClassDefinition), decl
        return qualify(
            decl.module_name,
            decl.short_name,
            prefix=ENCODED_CLASS_PROTO_NAME,
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
            paramtypes=[decl],
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

    def get_class_c_struct_field_name(defn: IR.FieldDefinition):
        assert not defn.extern, defn
        return encode(defn.name, prefix=ENCODED_CLASS_FIELD_PREFIX)

    def get_static_method_name(defn: IR.StaticMethodDefinition):
        assert isinstance(defn, IR.StaticMethodDefinition), defn
        return encode(
            f'{defn.cls.module_name}.{defn.cls.short_name}#{defn.name}',
            prefix=ENCODED_STATIC_METHOD_PREFIX,
        )

    # Get the C name of variable with given declaration
    cvarname = Multimethod('cvarname')

    @cvarname.on(IR.GlobalVariableDeclaration)
    def cvarname(self):
        return (
            self.short_name if self.extern else
            qualify(
                self.module_name,
                self.short_name,
                ENCODED_GLOBAL_VARIABLE_PREFIX,
            )
        )

    @cvarname.on(IR.Parameter)
    def cvarname(self):
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

    declare = Multimethod('declare')

    @declare.on(IR.PrimitiveTypeDeclaration)
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
                [OUTPUT_PTR_NAME] + list(pnames)
            )
            paramtypes = [
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

    def declare_raw_c_functype(self, name, pnames=None):
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

    @D.on(IR.GlobalVariableDefinition)
    def D(self, ctx):
        ctx.out += proto_for(self) + ';'

    @D.on(IR.PrimitiveTypeDefinition)
    def D(self, ctx):
        # These definitions are just to help the compiler,
        # nothing actually needs to get emitted for this.
        pass

    def emit_function_body(*, ctx, defn: IR.FunctionLike):
        assert defn.body, defn
        ctx.out += proto_for(defn)
        with ctx.retain_scope() as after_release:
            ctx.declare(ERROR_POINTER_TYPE, ERROR_POINTER_NAME)
            retvar = E(defn.body, ctx)
            if defn.rtype != IR.VOID:
                ctx.retain(retvar)
            if defn.extern:
                if defn.rtype != IR.VOID:
                    after_release += f'return {retvar};'
            else:
                if defn.rtype != IR.VOID:
                    after_release += f'*{OUTPUT_PTR_NAME} = {retvar};'
                after_release += f'return {ERROR_POINTER_NAME};'


    @D.on(IR.FunctionDefinition)
    def D(self, ctx):
        if self.body is not None:
            emit_function_body(ctx=ctx, defn=self)

    @D.on(IR.StructDefinition)
    def D(self, ctx):
        pass

    @D.on(IR.ClassDefinition)
    def D(self, ctx):
        struct_name = get_class_struct_name(self)
        malloc_name = get_class_malloc_name(self)
        malloc_type = get_class_malloc_type(self)
        delete_hook_name = get_class_delete_hook_name(self)
        delete_hook_type = get_delete_hook_type(self)
        deleter_name = get_class_deleter_name(self)
        proto_name = get_class_proto_name(self)

        ctx.static_fdecls += f'static {declare(malloc_type, malloc_name)};'
        ctx.static_fdecls += f'static {declare(DELETER_TYPE, deleter_name)};'

        ctx.out += f'KLC_Class {proto_name} = ' '{'
        with ctx.push_indent(1):
            ctx.out += f'"{self.module_name}",'
            ctx.out += f'"{self.short_name}",'
            ctx.out += f'&{deleter_name},'
        ctx.out += '};'

        ctx.out += 'static ' + declare(malloc_type, malloc_name)
        ctx.out += '{'
        with ctx.push_indent(1):
            ctx.out += f'{struct_name} zero = ' '{0};'
            ctx.out += f'{struct_name}* ret = malloc(sizeof({struct_name}));'
            ctx.out += f'*ret = zero;'
            ctx.out += f'ret->header.cls = &{proto_name};'
            ctx.out += f'return ret;'
        ctx.out += '}'

        ctx.out += 'static ' + declare(delete_hook_type, delete_hook_name, [
            THIS_NAME,
        ])
        with ctx.retain_scope():
            ctx.declare(ERROR_POINTER_TYPE, ERROR_POINTER_NAME)
            E(self.delete_hook.body, ctx)

        ctx.out += 'static ' + declare(DELETER_TYPE, deleter_name, ['robj', 'dq'])
        ctx.out += '{'
        retainable_fields = [
            field for field in self.fields if
            isinstance(field.type, IR.Retainable)
        ]
        with ctx.push_indent(1):
            ctx.out += f'{struct_name}* obj = ({struct_name}*) robj;'
            ctx.out += f'{delete_hook_name}(obj);'
            for field in retainable_fields:
                if isinstance(field.type, IR.ClassDefinition):
                    c_field_name = get_class_c_struct_field_name (field)
                    ctx.out += (
                        f'KLC_partial_release('
                        f'(KLC_Header*) obj->{c_field_name}, dq);'
                    )
                else:
                    assert False, field
        ctx.out += '}'

        for static_method in self.static_methods:
            D(static_method, ctx)

    @D.on(IR.StaticMethodDefinition)
    def D(self, ctx):
        emit_function_body(ctx=ctx, defn=self)

    proto_for = Multimethod('proto_for')

    @proto_for.on(IR.FunctionLike)
    def proto_for(self):
        return declare(
            self.type,
            cvarname(self),
            [
                encode(p.name, prefix=ENCODED_LOCAL_VARIABLE_PREFIX)
                for p in self.params
            ],
        )

    @proto_for.on(IR.GlobalVariableDefinition)
    def proto_for(self):
        decl = self.decl
        return declare(decl.type, cvarname(decl))

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

    @E.on(IR.IntLiteral)
    def E(self, ctx):
        retvar = ctx.declare(IR.PRIMITIVE_TYPE_MAP['int'])
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

    @E.on(IR.StringLiteral)
    def E(self, ctx):
        return f'"{escape_str(self.value)}"'

    def emit_function_call(
            *,
            ctx,
            rtype,
            args,
            to_extern,
            from_extern,
            c_function_name):
        argvars = ', '.join(E(arg, ctx) for arg in args)

        if to_extern:
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

            margvars = (
                f'{retvarp}, {argvars}' if args else retvarp
            )

            ctx.out += (
                f'{ERROR_POINTER_NAME} = {c_function_name}({margvars});'
            )
            ctx.out += f'if ({ERROR_POINTER_NAME}) ' '{'
            with ctx.push_indent(1):
                if from_extern:
                    ctx.out += f'KLC_panic_with_error({ERROR_POINTER_NAME});'
                else:
                    ctx.jump_out_of_scope()
            ctx.out += '}'
            return retvar

    @E.on(IR.FunctionCall)
    def E(self, ctx):
        fvar = E(self.f, ctx)
        return emit_function_call(
            ctx=ctx,
            rtype=self.type,
            args=self.args,
            to_extern=self.to_extern,
            from_extern=self.from_extern,
            c_function_name=fvar,
        )

    @E.on(IR.StaticMethodCall)
    def E(self, ctx):
        return emit_function_call(
            ctx=ctx,
            rtype=self.type,
            args=self.args,
            to_extern=False,
            from_extern=self.from_extern,
            c_function_name=get_static_method_name(self.f),
        )

    @E.on(IR.Malloc)
    def E(self, ctx):
        malloc_name = get_class_malloc_name(self.type)
        retvar = ctx.declare(self.type)
        ctx.out += f'{retvar} = {malloc_name}();'
        return retvar

    @E.on(IR.PointerCast)
    def E(self, ctx):
        exprvar = E(self.expr, ctx)
        retvar = ctx.declare(self.type)
        ctx.out += f'{retvar} = (({declare(self.type, "")}) {exprvar});'
        ctx.retain(retvar)
        return retvar

    @E.on(IR.ThrowStringLiteral)
    def E(self, ctx):
        escaped = escape_str(self.value)
        ctx.out += (
            f'{ERROR_POINTER_NAME} = '
            f'KLC_new_error_with_message("{escaped}");'
        )
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

    @init_expr_for.on(IR.PointerType)
    def init_expr_for(self):
        return 'NULL'

    @init_expr_for.on(IR.PrimitiveTypeDeclaration)
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

