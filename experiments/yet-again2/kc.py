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

    PRIMITIVE_TYPE_NAMES = (
        'void',
        'char',
        'short',
        'int',
        'long',
        'float',
        'double',
        'size_t',
    )
    ns(PRIMITIVE_TYPE_NAMES, 'PRIMITIVE_TYPE_NAMES')

    KEYWORDS = {
      'is', 'not', 'null', 'true', 'false', 'new', 'and', 'or', 'in',
      'inline', 'extern', 'class', 'trait', 'final', 'def', 'auto',
      'struct', 'const', 'throw',
      'for', 'if', 'else', 'while', 'break', 'continue', 'return',
      'with', 'from', 'import', 'as', 'try', 'catch', 'finally', 'raise',
      'except', 'case','switch', 'var',
    } | set(PRIMITIVE_TYPE_NAMES)
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

    @ns
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

        def format(self, depth=0, out=None):
            return_ = out is None
            out = [] if out is None else out
            ind = '  ' * depth
            out.append(f'{ind}{type(self).__name__}\n')
            for fname, _ in type(self).fields:
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
                for fieldname, _ in nt.fields
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
    class Type:
        """Marker class for what values indicate Types.
        """

        @property
        def is_primitive(self):
            return False

        @property
        def is_pointer(self):
            return False

    @ns
    class Declaration(Node):
        """Marker class for values that can be looked up by a Scope.

        abstract
            token: Token
        """

    @ns
    class VariableDeclaration(Declaration):
        pass

    @ns
    class Parameter(VariableDeclaration):
        fields = (
            ('type', Type),
            ('name', str),
        )

    @ns
    class FunctionDeclaration(Declaration):
        fields = (
            ('extern', bool),
            ('rtype', Type),
            ('module_name', str),
            ('short_name', str),
            ('params', typeutil.List[Parameter]),
            ('vararg', bool),
            ('has_body', bool),
        )

        @property
        def type(self):
            return FunctionType(
                self.extern,
                self.rtype,
                [p.type for p in self.params],
                self.vararg,
            )

    @ns
    class GlobalVariableDeclaration(VariableDeclaration):
        fields = (
            ('extern', bool),
            ('type', Type),
            ('module_name', str),
            ('short_name', str),
        )

    @ns
    class LocalVariableDeclaration(VariableDeclaration):
        fields = (
            ('type', Type),
            ('name', str),
        )

    @ns
    class TypeDeclaration(Type, Declaration):
        pass

    @ns
    class PrimitiveTypeDeclaration(TypeDeclaration):
        fields = (
            ('name', str),
        )

        @property
        def is_primitive(self):
            return True

    class ProxyMixin:
        def __eq__(self, other):
            return type(self) is type(other) and self._proxy == other._proxy

        def __hash__(self):
            return hash((type(self), self._proxy))

        def __repr__(self):
            return f'{type(self).__name__}{self._proxy}'

    @ns
    class RawPointerType(Type, ProxyMixin):
        def __init__(self, base):
            self.base = base

        @property
        def _proxy(self):
            return (self.base,)

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

    @ns
    class StructDeclaration(TypeDeclaration):
        defn = None
        fields = (
            ('module_name', str),
            ('short_name', str),
        )

        @property
        def extern(self):
            # Unfortunately, the 'extern' status of
            # a struct is not available at 'declaration' time.
            assert self.defn is not None, self
            return self.defn.extern

    PRIMITIVE_TYPE_MAP = {
        t: PrimitiveTypeDeclaration(builtin_token, t)
        for t in lexer.PRIMITIVE_TYPE_NAMES
    }
    ns(PRIMITIVE_TYPE_MAP, 'PRIMITIVE_TYPE_MAP')

    VOID = PRIMITIVE_TYPE_MAP['void']
    ns(VOID, 'VOID')

    VOIDP = RawPointerType(VOID)
    ns(VOIDP, 'VOIDP')

    EXCEPTION_POINTER = VOIDP
    ns(EXCEPTION_POINTER, 'EXCEPTION_POINTER')

    convertible = Multimethod('convertible', 2)
    ns(convertible, 'convertible')

    @convertible.on(PrimitiveTypeDeclaration, PrimitiveTypeDeclaration)
    def convertible(a, b):
        return a == b or (a.name, b.name) in {
            ('int', 'long'),
            ('int', 'size_t'),
            ('long', 'size_t'),
        }

    @convertible.on(Type, Type)
    def convertible(a, b):
        return a == b

    @ns
    class GlobalDefinition(Node):
        pass

    @ns
    class Expression(Node):
        """
        abstract
            type: Type
        """

    @ns
    class Block(CollectionNode):
        fields = (
            ('decls', typeutil.List[LocalVariableDeclaration]),
            ('exprs', typeutil.List[Expression]),
        )

        @property
        def type(self):
            return self.exprs[-1].type if self.exprs else VOID

    @ns
    class FunctionName(Expression):
        fields = (
            ('decl', FunctionDeclaration),
        )

        @property
        def type(self):
            return self.decl.type

    @ns
    class LocalName(Expression):
        fields = (
            ('decl', LocalVariableDeclaration),
        )

        @property
        def type(self):
            return self.decl.type

    @ns
    class SetLocalName(Expression):
        fields = (
            ('decl', LocalVariableDeclaration),
            ('expr', Expression),
        )

        @property
        def type(self):
            return self.decl.type

    @ns
    class StructField(Node):
        fields = (
            ('extern', bool),
            ('type', Type),
            ('name', str),
        )

    @ns
    class GetStructField(Expression):
        fields = (
            ('field_defn', StructField),
            ('expr', Expression),
        )

        @property
        def type(self):
            return self.field_defn.type

    @ns
    class SetStructField(Expression):
        fields = (
            ('field_defn', StructField),
            ('expr', Expression),
            ('valexpr', Expression),
        )

        @property
        def type(self):
            return self.field_defn.type

    @ns
    class FunctionCall(Expression):
        fields = (
            ('from_extern', bool),  # iff we are calling form extern func
            ('to_extern', bool),    # iff are are calling an extern function
            ('type', Type),
            ('f', Expression),
            ('args', typeutil.List[Expression]),
        )

    @ns
    class IntLiteral(Expression):
        type = PRIMITIVE_TYPE_MAP['int']

        fields = (
            ('value', int),
        )

    @ns
    class DoubleLiteral(Expression):
        type = PRIMITIVE_TYPE_MAP['double']

        fields = (
            ('value', float),
        )

    @ns
    class StringLiteral(Expression):
        type = RawPointerType(ConstType(PRIMITIVE_TYPE_MAP['char']))

        fields = (
            ('value', str),
        )

    @ns
    class ThrowStringLiteral(Expression):
        type = VOID

        fields = (
            ('value', str),
        )

    @ns
    class Include(Node):
        fields = (
            ('use_quotes', bool),
            ('value', str),
        )

    @ns
    class FromImport(Node):
        fields = (
            ('module_name', str),
            ('exported_name', str),
            ('alias', str),
        )

    @ns
    class StructDefinition(GlobalDefinition):
        fields = (
            ('extern', bool),
            ('decl', StructDeclaration),
            ('fields', typeutil.Optional[typeutil.List[StructField]]),
        )

    @ns
    class FunctionDefinition(GlobalDefinition):
        fields = (
            ('decl', FunctionDeclaration),
            ('body', typeutil.Optional[Block]),
        )

    @ns
    class GlobalVariableDefinition(GlobalDefinition):
        fields = (
            ('decl', GlobalVariableDeclaration),
        )

    @ns
    class Module(Node):
        fields = (
            ('name', str),
            ('includes', typeutil.List[Include]),
            ('imports', typeutil.List[FromImport]),
            ('definitions', typeutil.List[GlobalDefinition]),
        )


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

        def struct_type(self, token, module_name, name):
            if name not in self:
                self[name] = decl = (
                    IR.StructDeclaration(token, module_name, name)
                )
            decl = self[name]
            if not isinstance(decl, IR.StructDeclaration):
                with scope.push(token), scope.push(decl.token):
                    raise f'{name} is not a struct'
            return decl

        def __getitem__(self, key: str) -> IR.Declaration:
            if key in self.table:
                return self.table[key]
            elif self.parent is not None:
                return self.parent[key]
            else:
                raise self.error(f'{repr(key)} not defined')

        def __setitem__(self, key: str, value: IR.Declaration):
            assert isinstance(value, IR.Declaration), value
            if key in self.table:
                with self.push(value.token), self.push(self.table[key].token):
                    raise self.error(f'Duplicate definition of {repr(key)}')
            self.table[key] = value

        def __contains__(self, key: str) -> bool:
            return key in self.table or self.parent and key in self.parent

        def pull(self, key: str) -> IR.Declaration:
            if key not in self.table:
                if self.parent and key in self.parent:
                    self.table[key] = self.parent.pull(key)
                else:
                    raise self.error(f'{repr(key)} not defined')
            return self.table[key]

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
            with scope.push(token):
                exported_defn = (
                    scope.load_scope_for(module_name)[exported_name]
                )
            scope[alias] = exported_defn
            return Promise(lambda:
                IR.FromImport(token, module_name, exported_name, alias))

        def expect_name(name):
            if peek().type != 'NAME' or peek().value != name:
                with module_scope.push(peek()):
                    raise module_scope.error(f'Expected name {name}')
            return gettok()

        def expect_type(scope):
            token = peek()
            if token.type in IR.PRIMITIVE_TYPE_MAP:
                gettok()
                type = IR.PRIMITIVE_TYPE_MAP[token.type]
            else:
                name = expect_id()
                type = scope.struct_type(token, module_name, name)
            while True:
                token = peek()
                if consume('!'):
                    type = IR.RawPointerType(type)
                elif consume('*'):
                    type = IR.PointerType(type)
                elif consume('const'):
                    type = IR.ConstType(type)
                else:
                    break
            return type

        def expect_params(scope):
            expect('(')
            params = []
            vararg = False
            while not consume(')'):
                if consume('...'):
                    expect(')')
                    vararg = True
                    break
                ptok = peek()
                ptype = expect_type(scope)
                pname = expect_id()
                params.append(IR.Parameter(ptok, ptype, pname))
                if not consume(','):
                    expect(')')
                    break
            return params, vararg

        def at_variable_declaration():
            if peek().type in IR.PRIMITIVE_TYPE_MAP:
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

        def parse_block(parent_scope):
            scope = Scope(parent_scope)
            token = expect('{')
            decls = []
            exprs = []
            with skipping_newlines(False):
                consume_all('\n')
                while not consume('}'):
                    if at_variable_declaration():
                        dtoken = peek()
                        dtype = expect_type(scope)
                        dname = expect_id()
                        expr = (
                            parse_expression(scope) if consume('=') else None
                        )
                        expect('\n')
                        consume_all('\n')
                        decl = IR.LocalVariableDeclaration(
                            dtoken,
                            dtype,
                            dname,
                        )
                        decls.append(decl)
                        scope[dname] = decl
                        if expr is not None:
                            exprs.append(pcall(
                                IR.SetLocalName, dtoken, decl, expr))
                    else:
                        exprs.append(parse_expression(scope))
                    consume_all('\n')
            return Promise(lambda: IR.Block(
                token, decls, [p.resolve() for p in exprs]))

        def parse_expression(scope):
            return parse_postfix(scope)

        def parse_args(scope):
            argps = []
            expect('(')
            while not consume(')'):
                argps.append(parse_expression(scope))
                if not consume(','):
                    expect(')')
                    break
            return Promise(lambda: [p.resolve() for p in argps])

        def pfcall(scope, token, fp, argsp):
            @Promise
            def promise():
                f = fp.resolve()
                raw_args = argsp.resolve()
                if not isinstance(f.type, IR.FunctionType):
                    with scope.push(token):
                        raise scope.error(f'{f.type} is not a function')
                args = f.type.convert_args(scope, token, raw_args)
                return IR.FunctionCall(
                    token,
                    scope['@func'].extern,
                    f.type.extern,
                    f.type.rtype,
                    f,
                    args,
                )
            return promise

        def get_field_defn(scope, token, type, field_name):
            if not isinstance(type, IR.StructDeclaration):
                with scope.push(token):
                    raise scope.error(f'{type} is not a struct type')
            defn = type.defn
            fields = [f for f in defn.fields if f.name == field_name]
            if not fields:
                with scope.push(token):
                    raise scope.error(
                        f'{field_name} is not a member of {type}')
            field, = fields
            return field

        def pgetfield(scope, token, exprp, fname):
            @Promise
            def promise():
                expr = exprp.resolve()
                defn = get_field_defn(scope, token, expr.type, fname)
                return IR.GetStructField(
                    token,
                    defn,
                    expr,
                )
            return promise

        def psetfield(scope, token, exprp, fname, valp):
            @Promise
            def promise():
                expr = exprp.resolve()
                defn = get_field_defn(scope, token, expr.type, fname)
                val = scope.convert(valp.resolve(), defn.type)
                return IR.SetStructField(
                    token,
                    defn,
                    expr,
                    val,
                )
            return promise

        def parse_postfix(scope):
            expr = parse_primary(scope)
            while True:
                token = peek()
                if at('('):
                    argsp = parse_args(scope)
                    expr = pfcall(scope, token, expr, argsp)
                elif consume('.'):
                    name = expect_id()
                    if consume('='):
                        valp = parse_expression(scope)
                        expr = psetfield(scope, token, expr, name, valp)
                    else:
                        expr = pgetfield(scope, token, expr, name)
                else:
                    break
            return expr

        def pname(scope, token, name):
            @Promise
            def promise():
                with scope.push(token):
                    decl = scope[name]
                if isinstance(decl, IR.FunctionDeclaration):
                    return IR.FunctionName(token, decl)
                if isinstance(decl, IR.LocalVariableDeclaration):
                    return IR.LocalName(token, decl)
                with scope.push(token):
                    raise scope.fubar(f'{decl}')
            return promise

        def pthrowstrlit(scope, token, value):
            @Promise
            def promise():
                if scope['@func'].extern:
                    with scope.push(token):
                        raise scope.error(
                            f'You cannot throw from an extern function')
                return IR.ThrowStringLiteral(token, value)
            return promise

        def parse_primary(scope):
            token = peek()
            if consume('('):
                with skipping_newlines(True):
                    expr = parse_expression(scope)
                    return expr
            if consume('NAME'):
                name = token.value
                return pname(scope, token, name)
            if consume('INT'):
                return Promise.value(IR.IntLiteral(token, token.value))
            if consume('FLOAT'):
                return Promise.value(IR.DoubleLiteral(token, token.value))
            if consume('STRING'):
                return Promise.value(IR.StringLiteral(token, token.value))
            if consume('throw'):
                # For now, only allow string literals
                value = expect('STRING').value
                return pthrowstrlit(scope, token, value)
            with scope.push(peek()):
                raise scope.error(f'Expected expression but got {peek()}')

        def promfunc(scope, token, decl, bodyp):
            @Promise
            def promise():
                raw_body = bodyp.resolve() if bodyp else None
                if raw_body is not None and decl.rtype != IR.VOID:
                    body = scope.convert(raw_body, decl.rtype)
                else:
                    body = raw_body

                return IR.FunctionDefinition(token, decl, body)
            return promise

        def parse_struct(scope, token, extern):
            name = expect_id()
            decl = scope.struct_type(token, module_name, name)
            if scope[name] is not decl:
                scope[name] = decl

            fields = []
            expect('{')
            consume_all('\n')
            while not consume('}'):
                field_token = peek()
                field_extern = bool(consume('extern'))
                field_type = expect_type(scope)
                field_name = expect_id()
                fields.append(IR.StructField(
                    field_token,
                    field_extern or extern,
                    field_type,
                    field_name,
                ))
                consume_all('\n')

            @Promise
            def promise():
                for field in fields:
                    if isinstance(field.type, IR.StructDeclaration):
                        if field.type.defn is None:
                            with scope.push(field.token):
                                raise scope.error(
                                    f'Type {field.type} used as struct '
                                    f'member before being defined'
                                )
                decl.defn = IR.StructDefinition(
                    token,
                    extern,
                    decl,
                    fields,
                )
                return decl.defn
            return promise

        def parse_global(scope):
            token = peek()
            extern = bool(consume('extern'))
            if consume('struct'):
                return parse_struct(scope, token, extern)
            else:
                type = expect_type(scope)
                name = expect_id()
                if consume('\n'):
                    scope[name] = decl = IR.GlobalVariableDeclaration(
                        token,
                        extern,
                        type,
                        module_name,
                        name,
                    )
                    return Promise(lambda: IR.GlobalVariableDefinition(
                        token,
                        decl,
                    ))
                else:
                    params, vararg = expect_params(scope)
                    has_body = not consume('\n')
                    scope[name] = decl = IR.FunctionDeclaration(
                        token,
                        extern,
                        type,
                        module_name,
                        name,
                        params,
                        vararg,
                        has_body
                    )
                    fscope = Scope(scope)
                    fscope['@func'] = decl
                    bodyp = parse_block(fscope) if has_body else None
                    return promfunc(scope, token, decl, bodyp)

        module_token = peek()
        includes = []
        importps = []
        defnps = []

        consume_all('\n')
        while at('#'):
            itoken = expect('#')
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
            consume_all('\n')
            includes.append(IR.Include(itoken, use_quotes, ivalue))

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

    ENCODED_NAME_PREFIX = 'KLCN'

    ENCODED_FIELD_PREFIX = 'KLCF'

    OUT_VAR_NAME = 'KLCoutptr'

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

    def encode(name, prefix=ENCODED_NAME_PREFIX):
        return prefix + ''.join(map(encode_char, name))

    def relative_header_path_from_name(module_name):
        return module_name.replace('.', os.path.sep) + '.k.h'

    def relative_source_path_from_name(module_name):
        return module_name.replace('.', os.path.sep) + '.k.c'

    class TranslationUnit(IR.Node):
        fields = (
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

            # A place where temporary variables may be declared
            # We need this mechanism because in C89, you can't
            # willy-nilly declare variables anywhere (they
            # have to be declared at the beginning of a block).
            self._decls = None

        def new_temp_var_name(self):
            name = f'KLCT{self.next_temp_var_id}'
            self.next_temp_var_id += 1
            return name

        def declare(self, t: IR.Type, cname=None):
            cname = (
                self.new_temp_var_name() if cname is None else
                cname
            )
            self._decls += f'{declare(t, cname)} = {init_expr_for(t)};'
            return cname

        @contextlib.contextmanager
        def retain_scope(self):
            old_decls = self._decls
            out = self.out
            self._decls = out.spawn()
            try:
                yield
            finally:
                self._decls = old_decls

        @contextlib.contextmanager
        def push_indent(self, depth):
            old_out = self.out
            self.out = old_out.spawn(depth)
            try:
                yield
            finally:
                self.out = old_out

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
            elif isinstance(defn, IR.StructDefinition):
                cname = c_struct_name(defn.decl)
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
                            c_struct_field_name(field),
                        ) + ';';
                    if not defn.fields:
                        # Empty structs aren't standard C.
                        # Add a dummy field.
                        fields_out += f'char dummy;'
                    structs += '};'
            else:
                raise Fubar([], defn)

        return str(msb)

    def translate_source(module: IR.Module) -> str:
        ctx = Context()
        ctx.out += f'#include "{relative_header_path_from_name(module.name)}"'
        for defn in module.definitions:
            D(defn, ctx)
        return str(ctx.out)

    def qualify(module_name, name):
        return (
            encode(name) if module_name is None else
            encode(f'{module_name}.{name}')
        )

    def c_struct_name(decl: IR.StructDeclaration):
        return (
            decl.short_name if decl.extern else
            qualify(decl.module_name, decl.short_name)
        )

    def c_struct_field_name(defn: IR.StructField):
        return (
            defn.name if defn.extern else
            encode(defn.name, prefix=ENCODED_FIELD_PREFIX)
        )

    # Get the C name of variable with given declaration
    cvarname = Multimethod('cvarname')

    @cvarname.on(IR.GlobalVariableDeclaration)
    def cvarname(self):
        return (
            self.short_name if self.extern else
            qualify(self.module_name, self.short_name)
        )

    @cvarname.on(IR.LocalVariableDeclaration)
    def cvarname(self):
        return encode(self.name)

    @cvarname.on(IR.FunctionDeclaration)
    def cvarname(self):
        return (
            self.short_name if self.extern else
            qualify(self.module_name, self.short_name)
        )

    declare = Multimethod('declare')

    @declare.on(IR.PrimitiveTypeDeclaration)
    def declare(self, name):
        return f'{self.name} {name}'.strip()

    @declare.on(IR.StructDeclaration)
    def declare(self, name):
        cname = c_struct_name(self)
        return f'{cname} {name}'.strip()

    @declare.on(IR.RawPointerType)
    def declare(self, name):
        return declare(self.base, f' *{name}'.strip())

    @declare.on(IR.PointerType)
    def declare(self, name):
        raise Error([], [self, name])
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
            rtype = IR.EXCEPTION_POINTER
            pnames = None if pnames is None else (
                [OUT_VAR_NAME] + list(pnames)
            )
            paramtypes = [
                IR.RawPointerType(self.rtype)
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

    @D.on(IR.StructDefinition)
    def D(self, ctx):
        pass

    @D.on(IR.FunctionDefinition)
    def D(self, ctx):
        decl = self.decl

        if self.body is not None:
            ctx.out += proto_for(self) + ' {'
            with ctx.push_indent(1), ctx.retain_scope():
                retvar = E(self.body, ctx)
                if decl.extern:
                    if decl.rtype != IR.VOID:
                        ctx.out += f'return {retvar};'
                else:
                    if decl.rtype != IR.VOID:
                        ctx.out += f'*{OUT_VAR_NAME} = {retvar};'
                    ctx.out += f'return NULL;'
            ctx.out += '}'

    proto_for = Multimethod('proto_for')

    @proto_for.on(IR.FunctionDefinition)
    def proto_for(self):
        decl = self.decl
        return declare(
            decl.type,
            cvarname(decl),
            [encode(p.name) for p in decl.params],
        )

    @proto_for.on(IR.GlobalVariableDefinition)
    def proto_for(self):
        decl = self.decl
        return declare(decl.type, cvarname(decl))

    # translate expressions
    E = Multimethod('E')

    @E.on(IR.Block)
    def E(self, ctx):
        retvar = (
            ctx.declare(self.type) if self.type != IR.VOID else
            None
        )
        ctx.out += '{'
        last_retvar = None
        with ctx.push_indent(1), ctx.retain_scope():
            for decl in self.decls:
                ctx.declare(decl.type, cvarname(decl))
            for expr in self.exprs:
                last_retvar = E(expr, ctx)
            if self.type != IR.VOID:
                assert last_retvar is not None, self.type
                ctx.out += f'{retvar} = {last_retvar};'
        ctx.out += '}'
        return retvar

    @E.on(IR.SetLocalName)
    def E(self, ctx):
        cname = cvarname(self.decl)
        retvar = E(self.expr, ctx)
        ctx.out += f'{cname} = {retvar};'
        return cname

    @E.on(IR.LocalName)
    def E(self, ctx):
        return cvarname(self.decl)

    @E.on(IR.GetStructField)
    def E(self, ctx):
        # NOTE: It's kind of inefficient the way do it here
        # because of the copying we do with structs.
        structvar = E(self.expr, ctx)
        retvar = ctx.declare(self.type)
        cfname = c_struct_field_name(self.field_defn)
        ctx.out += f'{retvar} = {structvar}.{cfname};'
        return retvar

    @E.on(IR.SetStructField)
    def E(self, ctx):
        # NOTE: It's kind of inefficient the way do it here
        # because of the copying we do with structs.
        structvar = E(self.expr, ctx)
        resultvar = E(self.valexpr, ctx)
        retvar = ctx.declare(self.type)
        cfname = c_struct_field_name(self.field_defn)
        ctx.out += f'{retvar} = {resultvar};'
        ctx.out += f'{structvar}.{cfname} = {retvar};'
        return retvar

    @E.on(IR.IntLiteral)
    def E(self, ctx):
        return str(self.value)

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

    @E.on(IR.FunctionCall)
    def E(self, ctx):
        fvar = E(self.f, ctx)
        argvars = ', '.join(E(arg, ctx) for arg in self.args)

        if self.to_extern:
            if self.type == IR.VOID:
                ctx.out += f'{fvar}({argvars});'
                return
            else:
                retvar = ctx.declare(self.type)
                ctx.out += f'{retvar} = {fvar}({argvars});'
                return retvar
        else:
            if self.type == IR.VOID:
                retvar = None
                retvarp = 'NULL'
            else:
                retvar = ctx.declare(self.type)
                retvarp = f'&{retvar}'

            margvars = (
                f'{retvarp}, {argvars}' if self.args else retvarp
            )

            statvar = ctx.declare(IR.EXCEPTION_POINTER)
            ctx.out += f'{statvar} = {fvar}({margvars});'
            ctx.out += f'if ({statvar}) ' '{'
            with ctx.push_indent(1):
                if self.from_extern:
                    ctx.out += f'KLC_panic_with_error({statvar});'
                else:
                    ctx.out += f'return {statvar};'
            ctx.out += '}'
            return retvar

    @E.on(IR.ThrowStringLiteral)
    def E(self, ctx):
        escaped = escape_str(self.value)
        ctx.out += f'return KLC_new_error_with_message("{escaped}");'

    @E.on(IR.FunctionName)
    def E(self, ctx):
        return cvarname(self.decl)

    # For initializing variables when declaring them
    init_expr_for = Multimethod('init_expr_for')

    @init_expr_for.on(IR.StructDeclaration)
    def init_expr_for(self):
        return '{0}'

    @init_expr_for.on(IR.PointerType)
    def init_expr_for(self):
        return 'NULL'

    @init_expr_for.on(IR.RawPointerType)
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

