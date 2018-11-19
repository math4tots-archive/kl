import abc
import argparse
import contextlib
import itertools
import os
import re
import string
import subprocess
import sys
import typing

_scriptdir = os.path.dirname(os.path.realpath(__file__))

_special_method_prefixes = [
    'GET',
    'SET',
]

_special_method_names = {
    'Call',
    'Contains',
    'Iterator',
    'Next',
    'HasNext',
    'Add',
    'Sub',
    'Mul',
    'Div',
    'Eq',
    'Lt',
    'Str',
    'Repr',
    'Bool',
    'GetItem',
    'SetItem',
    'Slice',
    'SliceAll',
    'SliceLeft',
    'SliceRight',
    'SetSlice',
    'SetSliceAll',
    'SetSliceLeft',
    'SetSliceRight',
    'HashCode',
    'Enter',
    'Exit',
    'And',
    'Or',
    'Xor',
    'Lshift',
    'Rshift',
    'Invert',
}


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
                raise KeyError(types)
        return self.table[types]

    def __call__(self, *args):
        types = tuple(type(arg) for arg in args[:self.n])
        f = self.find(types)
        return f(*args)


PRIMITIVE_TYPES = {
    'void',
    'bool',
    'int',
    'double',
    'function',
    'type',
}

_primitive_method_names = {
    'null': [
        'Eq', 'HashCode',
        'Repr', 'Bool',
    ],
    'bool': [
        'Eq', 'Lt', 'HashCode',
        'Repr', 'Bool',
    ],
    'int': [
        'Eq', 'Lt', 'HashCode',
        'Add', 'Sub', 'Mul', 'Div', 'Mod', 'Pow',
        'Repr', 'Bool',
        'Xor', 'And', 'Or', 'Lshift', 'Rshift', 'Invert',
    ],
    'double': [
        'Eq', 'Lt', 'HashCode',
        'Add', 'Sub', 'Mul', 'Div',
        'Repr', 'Bool',
    ],
    'function': [
        'Eq', 'HashCode',
        'GETname', 'Repr', 'Bool',
    ],
    'type': [
        'Eq', 'HashCode',
        'GETname', 'Repr', 'Bool',
    ],
}

def nullable(type_):
    return type_ not in ('void', 'bool', 'int', 'double')

with open(os.path.join(_scriptdir, 'c', 'klc_prelude.c')) as f:
    CPRELUDE = f.read()

with open(os.path.join(_scriptdir, 'builtins.k')) as f:
    BUILTINS = f.read()


_safe_chars = set(
    string.ascii_lowercase +
    string.ascii_uppercase +
    ''.join(map(str, range(10)))) - {'Z'}


def encode(name):
    return 'KLCN' + encode0(name)


def encode0(name):
    """Encode a name so that it's safe to use as a C symbol

    Characters allowed in name:
        digits 0-9
        lower case letters a-z
        upper case letters A-Z
        underscore _
        special characters:
            dot          (.)
            dollar sign  ($)
            percent      (%)
            hash         (#) -> for <main> (just '#' by itself),
                                global variable init function names (#init),
                                'new' function (#new)
                                temporary variables (#tempvar#i)
            colon        (:) -> for method names <class>:<method>


    The special characters are to be used by the compiler for special
    generated names (e.g. auto-generated functions).

    The encoding mostly allows letters and digits to be themselves,
    except capital 'Z' is used as an escape character, to encode all
    other kinds of characters.
    """
    chars = []
    for c in name:
        if c == 'Z':
            chars.append('ZZ')
        elif c == '_':
            chars.append('ZA')
        elif c == '.':
            chars.append('ZB')
        elif c == '$':
            chars.append('ZC')
        elif c == '%':
            chars.append('ZD')
        elif c == '#':
            chars.append('ZE')
        elif c == ':':
            chars.append('ZF')
        elif c in _safe_chars:
            chars.append(c)
        else:
            raise Error([], f'Invalid character {c} in name {name}')
    return ''.join(chars)


class InverseSet:
    def __init__(self, items):
        self._items = frozenset(items)

    def __contains__(self, key):
        return key not in self._items

    def __repr__(self):
        return f'InverseSet({self._items})'


class Source(typing.NamedTuple):
    filename: str
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
        self.tokens = tokens
        self.message = message


@Namespace
def lexer(ns):
    KEYWORDS = {
      'is', 'not', 'null', 'true', 'false', 'new', 'and', 'or', 'in',
      'inline', 'extern', 'class', 'trait', 'final', 'def', 'auto',
      'for', 'if', 'else', 'while', 'break', 'continue', 'return',
      'with', 'from', 'import', 'as', 'try', 'catch', 'finally', 'raise',
      'except',
    }
    ns(KEYWORDS, 'KEYWORDS')

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


@Namespace
def ir(ns):
    "intermediate representation"

    @ns
    class Node(object):
        def __init__(self, token, *args):
            self.token = token
            for (fname, ftype), arg in zip(type(self).fields, args):
                # sys.stderr.write('ftype = %r\n' % (ftype, ))
                if not isinstance(arg, ftype):
                    raise TypeError('Expected type of %r to be %r, but got %r' % (
                        fname, ftype, arg))
                try:
                    setattr(self, fname, arg)
                except AttributeError:
                    sys.stderr.write(f'fname = {fname}\n')
                    raise
            if len(type(self).fields) != len(args):
                raise TypeError('%s expects %s arguments, but got %s' % (
                    type(self).__name__, len(type(self).fields), len(args)))

        def __repr__(self):
            return '%s(%s)' % (
                type(self).__name__,
                ', '.join(repr(getattr(self, n)) for n, _ in type(self).fields),
            )

    @ns
    class BaseVariableDefinition(Node):
        fields = (
            ('type', str),
            ('name', str),
        )

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.type == 'void':
                raise Error([self.token], f'Variable of type void is not allowed')
            if self.type == 'Closure':
                raise Error(
                    [self.token],
                    f'Variable of type Closure is not allowed (use var instead)')

    @ns
    class Parameter(BaseVariableDefinition):
        pass

    @ns
    class Field(BaseVariableDefinition):
        pass

    @ns
    class GlobalDefinition(Node):
        pass

    @ns
    class Statement(Node):
        # def translate(self, ctx):
        #     """Translates self with the given translation context.
        #     Returns True if the current statement is terminal (e.g. return).
        #     No statements directly after a translate that returns True should
        #     be emitted, since those following statements will be unreachable.
        #     """
        #     raise NotImplementedError()
        pass

    @ns
    class Expression(Node):
        # def translate(self, ctx):
        #     """Translates self with the given translation context.
        #     Returns (type, tempvar) pair, where type is the type of
        #     evaluating this expression, and tempvar is the name of the
        #     temporary C variable where the result of this expression
        #     is stored.
        #     If type is 'void', tempvar should be None.
        #     """
        #     raise NotImplementedError()
        pass

    @ns
    class Name(Expression):
        fields = (
            ('name', str),
        )

    @ns
    class SetName(Expression):
        fields = (
            ('name', str),
            ('expression', Expression),
        )

    @ns
    class Cast(Expression):
        fields = (
            ('expression', Expression),
            ('type', str),
        )

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if isinstance(self.expression, NullLiteral):
                raise Error(
                    [self.token],
                    'Type assertions on null values will almost always fail '
                    '(try a typed null instead, '
                    'e.g. null(String) without the period)')

    @ns
    class NullLiteral(Expression):
        fields = (
            ('type', str),
        )

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not nullable(self.type):
                raise Error([self.token], f'{self.type} is not nullable')

    @ns
    class BoolLiteral(Expression):
        fields = (
            ('value', bool),
        )

    @ns
    class StringLiteral(Expression):
        fields = (
            ('value', str),
        )

    @ns
    class IntLiteral(Expression):
        fields = (
            ('value', int),
        )

    @ns
    class DoubleLiteral(Expression):
        fields = (
            ('value', float),
        )

    @ns
    class ListDisplay(Expression):
        fields = (
            ('expressions', List[Expression]),
        )

    @ns
    class BinaryComparison(Expression):
        """Base class for binary operations that return bool result.
        """
        fields = (
            ('left', Expression),
            ('right', Expression),
        )

        method_types = set()
        fallback_method = None

        # The operator to use to compare the two values if
        # the either of the expressions are null
        op_for_null = None

        # Customizations for when using a fallback method.
        reverse_method_args = False
        invert_method_result = False

    @ns
    class Is(BinaryComparison):
        fast_op_types = InverseSet({'var'})
        fast_op = '=='
        fallback_fn = '_Is'

    @ns
    class IsNot(BinaryComparison):
        fast_op_types = InverseSet({'var'})
        fast_op = '!='
        fallback_fn = '_IsNot'

    @ns
    class Equals(BinaryComparison):
        fast_op_types = frozenset(PRIMITIVE_TYPES)
        fast_op = '=='
        op_for_null = '=='
        fallback_method = 'Eq'
        fallback_fn = '_Eq'

    @ns
    class NotEquals(BinaryComparison):
        fast_op_types = frozenset(PRIMITIVE_TYPES)
        fast_op = '!='
        op_for_null = '!='
        fallback_method = 'Eq'
        invert_method_result = True
        fallback_fn = '_Ne'

    @ns
    class LessThan(BinaryComparison):
        fast_op_types = ['int', 'double']
        fast_op = '<'
        fallback_method = 'Lt'
        fallback_fn = '_Lt'

    @ns
    class LessThanOrEqual(BinaryComparison):
        fast_op_types = ['int', 'double']
        fast_op = '<='
        fallback_method = 'Lt'
        invert_method_result = True
        reverse_method_args = True
        fallback_fn = '_Le'

    @ns
    class GreaterThan(BinaryComparison):
        fast_op_types = ['int', 'double']
        fast_op = '>'
        fallback_method = 'Lt'
        reverse_method_args = True
        fallback_fn = '_Gt'

    @ns
    class GreaterThanOrEqual(BinaryComparison):
        fast_op_types = ['int', 'double']
        fast_op = '>='
        fallback_method = 'Lt'
        invert_method_result = True
        fallback_fn = '_Ge'

    @ns
    class FunctionCall(Expression):
        fields = (
            ('function', str),
            ('args', List[Expression]),
        )

    @ns
    class MethodCall(Expression):
        fields = (
            ('owner', Expression),
            ('name', str),
            ('args', List[Expression]),
        )

    @ns
    class BaseInOperation(Expression):
        fields = (
            ('left', Expression),
            ('right', Expression),
        )

        method_name = None

    @ns
    class In(BaseInOperation):
        method_name = 'Contains'

    @ns
    class LogicalNot(Expression):
        fields = (
            ('expression', Expression),
        )

    @ns
    class LogicalOr(Expression):
        fields = (
            ('left', Expression),
            ('right', Expression),
        )

    @ns
    class LogicalAnd(Expression):
        fields = (
            ('left', Expression),
            ('right', Expression),
        )

    @ns
    class Conditional(Expression):
        fields = (
            ('condition', Expression),
            ('left', Expression),
            ('right', Expression),
        )

    @ns
    class VariableDefinition(Statement, BaseVariableDefinition):
        fields = (
            ('final', bool),
            ('type', Optional[str]),
            ('name', str),
            ('expression', Optional[Expression]),
        )

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.type is None and self.expression is None:
                raise Error(
                    [self.token],
                    'If explicit type is not specified for '
                    'a variable definition, an expression must be supplied')

    @ns
    class ExpressionStatement(Statement):
        fields = (
            ('expression', Optional[Expression]),
        )

    @ns
    class Return(Statement):
        fields = (
            ('expression', Optional[Expression]),
        )

    @ns
    class Block(Statement):
        fields = (
            ('statements', List[Statement]),
        )

    @ns
    class If(Statement):
        fields = (
            ('condition', Expression),
            ('body', Block),
            ('other', Optional[Statement]),
        )

        def __init__(self, *args ,**kwargs):
            super().__init__(*args, **kwargs)
            if not isinstance(self.other, (type(None), If, Block)):
                raise Error(
                    [self.other.token],
                    f"Expected block or if statement")

    @ns
    class While(Statement):
        fields = (
            ('condition', Expression),
            ('body', Block),
        )

    @ns
    class Program(Node):
        fields = (
            ('definitions', List[GlobalDefinition]),
            ('env', dict),
        )

    @ns
    class GlobalVariableDefinition(GlobalDefinition):
        fields = (
            ('extern', bool),
            ('type', str),
            ('name', str),
        )

    @ns
    class FunctionDefinition(GlobalDefinition):
        fields = (
            ('return_type', str),
            ('name', str),
            ('params', List[Parameter]),
            ('body', Optional[Block]),
        )

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for m in ('Eq', 'Lt'):
                if self.name.endswith(f':{m}'):
                    if self.return_type != 'bool':
                        raise Error([self.token],
                                    f'{m} methods must return bool')
            if self.name == 'main':
                if self.return_type != 'void' or len(self.params):
                    raise Error([self.token],
                                'main function must have signature '
                                "'void main()'")

        @property
        def extern(self):
            return self.body is None

    @ns
    class TypeDefinition(GlobalDefinition):
        pass

    @ns
    class TraitDefinition(GlobalDefinition):
        # Trait kind of defines a type, but TraitDefinition not inheriting
        # TypeDefinition is on purpose. Right now, there's no real support
        # for declaring a variable as a trait type.
        fields = (
            ('name', str),
            ('traits', List[str]),
            ('methods', List[str]),
        )

        _trait_method_map = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    @ns
    class ClassDefinition(TypeDefinition):
        fields = (
            ('name', str),
            ('traits', List[str]),
            ('fields', Optional[List[Field]]),
            ('methods', List[str]),
            ('untyped_methods', List[str]),
        )

        _method_map = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.untyped_methods and self.name != 'Closure':
                raise Error(
                    [self.token],
                    'Closure is the only type allowed to have untyped methods')

        @property
        def extern(self):
            return self.fields is None


@Namespace
def Cee(ns):
    @ns
    class Scope(object):
        def __init__(self, parent):
            self.parent = parent
            self.table = dict()
            self.pulled = set()  # set of names that were pulled from parent scope
            self.local_definitions = []

        def _set(self, name, node):
            if name in self.table:
                raise Error([self.table[name].token, node.token],
                            f'Name {name} conflict')
            self.table[name] = node

        def add(self, node):
            self.local_definitions.append(node)
            if isinstance(node, ir.GlobalDefinition):
                self._set(node.name, node)
            elif isinstance(node, ir.BaseVariableDefinition):
                self.validate_vardef(node)
                if isinstance(node, ir.VariableDefinition):
                    self._check_for_shadow(node)
                self._set(node.name, node)
            else:
                raise Error([node.token], f'FUBAR: Unrecognized node type {node}')

        def validate_vardef(self, vardef):
            if vardef.type not in PRIMITIVE_TYPES and vardef.type != 'var':
                typenode = self.get(vardef.type, [vardef.token])
                if isinstance(typenode, ir.TraitDefinition):
                    raise Error(
                        [vardef.token],
                        f'Using trait types variable type not supported')
                if not isinstance(typenode, ir.TypeDefinition):
                    raise Error([vardef.token], f'{vardef.type} is not a type')

        def _check_for_shadow(self, node):
            # We need to take care not to shadow VariableDefinitions.
            # If VariableDefinitions are ever shadowed, they will cause
            # incorrect behavior when the function tries to release
            # all local variables before a return.
            scope = self.parent
            name = node.name
            while scope:
                if name in scope.table and isinstance(scope.table[name], ir.VariableDefinition):
                    raise Error([node.token, scope.table[name].token],
                                f'Shadowed local variable')
                scope = scope.parent

        def _missing_name_err(self, name, tokens):
            return Error(tokens, f'Name {name} is not defined')

        def pull(self, name, tokens):
            """Pull a name from parent scope.
            The purpose of doing this is so that if an outer variable is
            used, then a user tries to declare a variable of the same name,
            we want to error out and tell the user that that's funky.
            """
            if name not in self.table:
                if self.parent is None:
                    raise self._missing_name_err(name, tokens)
                self.table[name] = self.parent.pull(name, tokens)
                self.pulled.add(name)
            return self.table[name]

        def get(self, name, tokens):
            if name in self.table:
                return self.table[name]
            elif self.parent is not None:
                return self.parent.get(name, tokens)
            raise self._missing_name_err(name, tokens)

    @ns
    class TranslationContext(object):
        def __init__(self, scope):
            self.scope = scope

        def cdecltype(self, name):
            if name in ('function', 'int', 'bool'):
                return f'KLC_{name}'
            elif name == 'type':
                return f'KLC_type'
            if name in PRIMITIVE_TYPES:
                return name
            elif name == 'var':
                return 'KLC_var'
            else:
                return f'{self.cname(name)}*'

        def cname(self, name):
            return encode(name)

        def czero(self, name):
            if name == 'type':
                return 'NULL'
            elif name in PRIMITIVE_TYPES:
                return '0'
            elif name == 'var':
                return 'KLC_null'
            else:
                return 'NULL'

        def varify(self, type_, cname):
            if type_ == 'var':
                return cname
            elif type_ in PRIMITIVE_TYPES:
                return f'KLC_{type_}_to_var({cname})'
            else:
                return f'KLC_object_to_var((KLC_header*) {cname})'

        def unvarify(self, type_, cname):
            if type_ == 'var':
                return cname
            elif type_ in PRIMITIVE_TYPES:
                return f'KLC_var_to_{type_}({cname})'
            else:
                return f'({self.cdecltype(type_)}) KLC_var_to_object({cname}, &KLC_type{encode0(type_)})'

    @ns
    class GlobalTranslationContext(TranslationContext):
        def __init__(self, program):
            super().__init__(Scope(None))
            self.gctx = self
            for d in program.definitions:
                self.scope.add(d)
            self.out = FractalStringBuilder(0)
            self.inc = self.out.spawn()
            self.out(CPRELUDE)
            self.fwd = self.out.spawn()
            self.hdr = self.out.spawn()
            self.src = self.out.spawn()

        def fctx(self, fdef):
            "Create child function translation context"
            return FunctionTranslationContext(self, fdef)

    @ns
    class BodyTranslationContext(TranslationContext):

        def bctx(self, depth):
            return BlockTranslationContext(self, depth)

    @ns
    class FunctionTranslationContext(BodyTranslationContext):
        def __init__(self, gctx, fdef):
            super().__init__(Scope(gctx.scope))
            for param in fdef.params:
                self.scope.add(param)
            self.parent = gctx
            self.gctx = gctx
            self.fdef = fdef
            self.hdr = gctx.hdr.spawn()
            self.src = gctx.src.spawn()
            self.fctx = self

    @ns
    class BlockTranslationContext(BodyTranslationContext):
        def __init__(self, parent: BodyTranslationContext, depth):
            super().__init__(Scope(parent.scope))
            self.parent = parent
            self.fctx = parent.fctx
            self.gctx = parent.gctx
            self.src = parent.src.spawn(depth)

        def ectx(self):
            return ExpressionTranslationContext(self)

    @ns
    class ExpressionTranslationContext(TranslationContext):
        def __init__(self, parent: BlockTranslationContext):
            super().__init__(parent.scope)
            self.parent = parent
            self.gctx = parent.gctx
            self.fctx = parent.fctx
            self.tempdecls = []  # (type, cname) pairs of temporary vars
            parent.src += '{'
            self.tmp = parent.src.spawn(1)
            self.src = parent.src.spawn(1)
            parent.src += '}'
            self.next_tempvar_index = 0

        def _next_tempvar_name(self):
            i = self.next_tempvar_index
            self.next_tempvar_index += 1
            return f'tempvar{i}'

        def mktemp(self, type_):
            tempvar = self._next_tempvar_name()
            self.tempdecls.append((type_, tempvar))
            self.tmp += f'{self.cdecltype(type_)} {tempvar} = {self.czero(type_)};'
            return tempvar

        def mktemparr(self, n):
            if n == 0:
                raise Error([], 'FUBAR: zero-length array')
            tempvar = self._next_tempvar_name()
            self.tmp += f'KLC_var {tempvar}[{n}];'
            return tempvar

        def release_tempvars(self, keepvar=None):
            """Release all temporary variables used during expression
            evaluation, except the specified keepvar.
            Variables are released in LIFO order
            """
            for type_, cname in reversed(self.tempdecls):
                if cname != keepvar:
                    self.src += _crelease(self, type_, cname)

    translate = Multimethod('translate')
    ns(translate, name='translate')

    @translate.on(ir.Name)
    def translate(self, ctx):
        defn = ctx.scope.get(self.name, [self.token])
        if isinstance(defn, ir.GlobalVariableDefinition):
            etype = defn.type
            tempvar = ctx.mktemp(etype)
            ctx.src += f'{tempvar} = KLC_get_global{encode(defn.name)}();'
            return (etype, tempvar)
        elif isinstance(defn, ir.BaseVariableDefinition):
            etype = defn.type
            tempvar = ctx.mktemp(etype)
            ctx.src += f'{tempvar} = {ctx.cname(self.name)};'
            ctx.src += _cretain(ctx, etype, tempvar)
            return (etype, tempvar)
        elif isinstance(defn, ir.ClassDefinition):
            tempvar = ctx.mktemp('type')
            ctx.src += f'{tempvar} = &KLC_type{encode0(defn.name)};'
            return 'type', tempvar
        elif isinstance(defn, ir.FunctionDefinition):
            tempvar = ctx.mktemp('function')
            ctx.src += f'{tempvar} = &KLC_functioninfo{encode(defn.name)};'
            return 'function', tempvar
        else:
            raise Error([self.token, defn.token],
                        f'{name} is not a variable')

    @translate.on(ir.SetName)
    def translate(self, ctx):
        defn = ctx.scope.get(self.name, [self.token])
        if isinstance(defn, ir.GlobalVariableDefinition):
            raise Error(
                [self.token, defn.token],
                f'Global variables are final ({defn.name})')
        elif isinstance(defn, ir.BaseVariableDefinition):
            etype = defn.type
            tempvar = ctx.mktemp(etype)
            etype, evar = translate(self.expression, ctx)
            ctx.src += _cretain(ctx, etype, evar)
            ctx.src += _defcrelease(defn, ctx)
            if etype != defn.type:
                if defn.type == 'var':
                    evar = ctx.varify(etype, evar)
                else:
                    raise Error([self.token, defn.token],
                                f'Tried to set {self.name} of type '
                                f'{defn.type} to {etype}')
            ctx.src += f'{ctx.cname(self.name)} = {evar};'
            return (etype, tempvar)
        else:
            raise Error([self.token, defn.token],
                        f'{name} is not a variable')

    @translate.on(ir.Cast)
    def translate(self, ctx):
        etype, etempvar = translate(self.expression, ctx)
        if etype != 'var':
            raise Error(
                [self.token],
                f'Only var types can be cast '
                f'into other types but got {etype}')
        tempvar = ctx.mktemp(self.type)
        with with_frame(ctx, self.token):
            ctx.src += f'{tempvar} = {ctx.unvarify(self.type, etempvar)};'
        ctx.src += _cretain(ctx, self.type, tempvar)
        return (self.type, tempvar)

    @translate.on(ir.NullLiteral)
    def translate(self, ctx):
        tempvar = ctx.mktemp(self.type)
        if self.type == 'var':
            ctx.src += f'{tempvar} = KLC_null;'
        else:
            ctx.src += f'{tempvar} = NULL;'
        return self.type, tempvar

    @translate.on(ir.BoolLiteral)
    def translate(self, ctx):
        tempvar = ctx.mktemp('bool')
        ctx.src += f'{tempvar} = {1 if self.value else 0};'
        return ('bool', tempvar)

    @translate.on(ir.StringLiteral)
    def translate(self, ctx):
        tempvar = ctx.mktemp('String')
        # TODO: properly escape the string literal
        s = (self.value
            .replace('\\', '\\\\')
            .replace('\t', '\\t')
            .replace('\n', '\\n')
            .replace('\r', '\\r')
            .replace('"', '\\"')
            .replace("'", "\\'"))
        ctx.src += f'{tempvar} = KLC_mkstr("{s}");'
        return ('String', tempvar)

    @translate.on(ir.IntLiteral)
    def translate(self, ctx):
        tempvar = ctx.mktemp('int')
        # TODO: Warn if size of 'value' is too big
        ctx.src += f'{tempvar} = {self.value}L;'
        return ('int', tempvar)

    @translate.on(ir.DoubleLiteral)
    def translate(self, ctx):
        tempvar = ctx.mktemp('double')
        # TODO: Warn if size of 'value' is too big
        ctx.src += f'{tempvar} = {self.value};'
        return ('double', tempvar)

    @translate.on(ir.ListDisplay)
    def translate(self, ctx):
        argvars = []
        for expr in self.expressions:
            etype, evar = translate(expr, ctx)
            argvars.append(ctx.varify(etype, evar))
        xvar = ctx.mktemp('List')
        ctx.src += f'{xvar} = KLC_mklist({len(argvars)});'
        for arg in argvars:
            ctx.src += f'{encode("List:push")}({xvar}, {arg});'
        return 'List', xvar

    @translate.on(ir.BinaryComparison)
    def translate(self, ctx):
        fast_op_types = type(self).fast_op_types
        fast_op = type(self).fast_op
        fallback_method = type(self).fallback_method
        fallback_fn = type(self).fallback_fn
        op_for_null = type(self).op_for_null
        reverse_method_args = type(self).reverse_method_args
        invert_method_result = type(self).invert_method_result

        ltok = self.left.token
        ltype, lvar = _translate_value(self.left, ctx)

        rtok = self.right.token
        rtype, rvar = _translate_value(self.right, ctx)

        # If the types are the same and not 'var', we might be able to
        # do some optimizations.
        if ltype == rtype and ltype != 'var':
            # If the types are in fast_op_types,
            # we can just run a single C operation to
            # compute the result
            if ltype in fast_op_types:
                xvar = ctx.mktemp('bool')
                ctx.src += f'{xvar} = ({lvar}) {fast_op} ({rvar});'
                return 'bool', xvar

            # If there is a fallback method to call,
            # try that. Since the types are the same and
            # the type is not var, a method call is really
            # a plain typed function call here, so would
            # be faster than a generic function call
            # like with the final fallback.
            if fallback_method:
                if reverse_method_args:
                    ltok, ltype, lvar, rtok, rtype, rvar = (
                        rtok, rtype, rvar, ltok, ltype, lvar)

                argtriples = [
                    (self.left.token, ltype, lvar),
                    (self.right.token, rtype, rvar),
                ]

                # null check, but we only care about this if:
                #  1. we're comparing nullable types, and
                #  2. the operator supports some kind of alternate
                #     behavior on null. In our case, in case of null
                #     we substitute with the 'op_for_null' operator
                #     if set.
                #  *** We also cheat a little bit here.
                #      Since we know that the only operators that specify
                #      op_for_null are Equals and NotEquals, we can
                #      also skip doing the full method call if
                #      left and right are pointer equal.
                if nullable(ltype) and op_for_null:
                    xvar0 = ctx.mktemp('bool')
                    ctx.src += f'if ({lvar} && {rvar} && {lvar} != {rvar}) ''{'

                xtype, xvar = _translate_mcall(
                    ctx, self.token, fallback_method, argtriples)
                assert xtype == 'bool', xtype
                if invert_method_result:
                    ctx.src += f'{xvar} = !{xvar};'

                if nullable(ltype) and op_for_null:
                    ctx.src += f'{xvar0} = {xvar};'
                    ctx.src += '} else {'
                    ctx.src += f'{xvar0} = {lvar} {op_for_null} {rvar};'
                    ctx.src += '}'
                    return 'bool', xvar0
                else:
                    return 'bool', xvar

        # If none of the above match, use the fallback function
        fn = ctx.scope.get(fallback_fn, [self.token])
        argtriples = [
            (self.left.token, ltype, lvar),
            (self.right.token, rtype, rvar),
        ]
        return _translate_fcall(ctx, self.token, fn, argtriples)

    @translate.on(ir.FunctionCall)
    def translate(self, ctx):
        defn = ctx.scope.get(self.function, [self.token])
        if isinstance(defn, ir.FunctionDefinition):
            argtriples = []
            for arg in self.args:
                argtype, argtempvar = translate(arg, ctx)
                argtriples.append((arg.token, argtype, argtempvar))
            return _translate_fcall(ctx, self.token, defn, argtriples)

        if (isinstance(defn, ir.BaseVariableDefinition) and
                defn.type in ('var', 'function')):
            argtempvars = []
            for arg in self.args:
                argtype, argtempvar = translate(arg, ctx)
                if argtype == 'void':
                    raise Error(
                        [arg.token],
                        'void expression cannot be used as an argument')
                argtempvar = ctx.varify(argtype, argtempvar)
                argtempvars.append(argtempvar)
            tempvar = ctx.mktemp('var')
            nargs = len(self.args)
            temparr = ctx.mktemparr(nargs) if nargs else 'NULL'
            cfname = ctx.cname(defn.name)
            for i, argtempvar in enumerate(argtempvars):
                ctx.src += f'{temparr}[{i}] = {argtempvar};'
            if defn.type == 'function':
                ctx.src += f'{tempvar} = {cfname}->body({nargs}, {temparr});'
            else:
                ctx.src += f'{tempvar} = KLC_var_call({cfname}, {nargs}, {temparr});'
            return 'var', tempvar

        if isinstance(defn, ir.BaseVariableDefinition):
            # At this point, we know defn.type cannot be var or function
            # For these types, we just want to call the 'Call' method.
            argtriples = [
                (self.token, defn.type, ctx.cname(defn.name)),
            ]
            for arg in self.args:
                argtype, argtempvar = _translate_value(arg, ctx)
                argtriples.append((arg.token, argtype, argtempvar))
            return _translate_mcall(ctx, self.token, 'Call', argtriples)

        if isinstance(defn, ir.ClassDefinition):
            argtriples = []
            for arg in self.args:
                argtype, argtempvar = translate(arg, ctx)
                argtriples.append((arg.token, argtype, argtempvar))
            malloc_name = f'KLC_malloc{encode0(defn.name)}'
            fname = f'{defn.name}#new'
            fdefn = ctx.scope.get(fname, [self.token])
            if not isinstance(fdefn, ir.FunctionDefinition):
                raise Error([self.token], f'FUBAR: shadowed function name')

            if defn.extern:
                # For extern types, constructing the type is
                # identical to a single function call
                return _translate_fcall(ctx, self.token, fdefn, argtriples)
            else:
                # For normal types, we want to malloc first, and then
                # call the initializer
                # TODO: Simplify normal types so that these are also
                # just a single function call like extern types.
                this_tempvar = ctx.mktemp(defn.name)
                ctx.src += f'{this_tempvar} = {malloc_name}();'
                argtriples = [
                    (self.token, defn.name, this_tempvar)
                ] + argtriples
                _translate_fcall(ctx, self.token, fdefn, argtriples)
                return defn.name, this_tempvar

        raise Error([self.token, defn.token],
                    f'{self.function} is not a function')

    @translate.on(ir.MethodCall)
    def translate(self, ctx):
        ownertype, ownertempvar = translate(self.owner, ctx)
        argtriples = [(self.token, ownertype, ownertempvar)]
        for arg in self.args:
            argtype, argvar = translate(arg, ctx)
            argtriples.append((arg.token, argtype, argvar))
        return _translate_mcall(ctx, self.token, self.name, argtriples)

    @translate.on(ir.BaseInOperation)
    def translate(self, ctx):
        ltype, lvar = translate(self.left, ctx)
        rtype, rvar = translate(self.right, ctx)
        # This could almost just be a simple MethodCall, but the problem
        # is that the order or arguments have to be reversed
        argtriples = [
            (self.right.token, rtype, rvar),
            (self.left.token, ltype, lvar),
        ]
        return _translate_mcall(
            ctx, self.token, type(self).method_name, argtriples)

    @translate.on(ir.LogicalNot)
    def translate(self, ctx):
        etype, evar = translate(self.expression, ctx)
        xvar = ctx.mktemp('bool')
        ctx.src += f'{xvar} = !{_ctruthy(ctx, etype, evar)};'
        return 'bool', xvar

    @translate.on(ir.LogicalOr)
    def translate(self, ctx):
        ltype, lvar = translate(self.left, ctx)
        xvar = ctx.mktemp('bool')
        ctx.src += f'if (!{_ctruthy(ctx, ltype, lvar)})'
        ctx.src += '{'
        rtype, rvar = translate(self.right, ctx)
        ctx.src += f'{xvar} = {_ctruthy(ctx, rtype, rvar)};'
        ctx.src += '} else {'
        ctx.src += f'{xvar} = 1;'
        ctx.src += '}'
        if 'void' in (ltype, rtype):
            raise Error([self.token], 'void type in or operator')
        return ('bool', xvar)

    @translate.on(ir.LogicalAnd)
    def translate(self, ctx):
        ltype, lvar = translate(self.left, ctx)
        xvar = ctx.mktemp('bool')
        ctx.src += f'if ({_ctruthy(ctx, ltype, lvar)})'
        ctx.src += '{'
        rtype, rvar = translate(self.right, ctx)
        ctx.src += f'{xvar} = {_ctruthy(ctx, rtype, rvar)};'
        ctx.src += '} else {'
        ctx.src += f'{xvar} = 0;'
        ctx.src += '}'
        if 'void' in (ltype, rtype):
            raise Erorr([self.token], 'void type in and operator')
        return ('bool', xvar)

    @translate.on(ir.Conditional)
    def translate(self, ctx):
        ctype, cvar = translate(self.condition, ctx)
        ctx.src += f'if ({_ctruthy(ctx, ctype, cvar)})'
        ctx.src += '{'
        ltype, lvar = translate(self.left, ctx)
        lsrc = ctx.src.spawn()
        ctx.src += '} else {'
        rtype, rvar = translate(self.right, ctx)
        rsrc = ctx.src.spawn()
        ctx.src += '}'
        if 'void' in (ltype, rtype):
            raise Error([self.token], 'void type in conditional operator')
        if 'var' in (ltype, rtype):
            xtype = 'var'
            lvar = ctx.varify(lvar)
            rvar = ctx.varify(rvar)
        elif ltype != rtype:
            raise Error(
                [self.token],
                'conditional operator requires result types to be the same')
        else:
            xtype = ltype

        xvar = ctx.mktemp(xtype)
        lsrc += f'{xvar} = {lvar};'
        rsrc += f'{xvar} = {rvar};'
        ctx.src += _cretain(ctx, xtype, xvar)
        return (xtype, xvar)

    @translate.on(ir.VariableDefinition)
    def translate(self, ctx):
        if self.expression:
            ectx = ctx.ectx()
            etype, tempvar = translate(self.expression, ectx)
            value = tempvar
            if self.type is None:
                self.type = etype
            elif self.type == 'var' and etype != 'void':
                value = ctx.varify(etype, tempvar)
            elif self.type != etype:
                raise Error([self.token],
                            f'Expected {self.type} but got {etype}')
            ectx.src += f'{_cname(self, ectx)} = {value};'
            ectx.release_tempvars(tempvar)

        ctx.scope.add(self)

    @translate.on(ir.ExpressionStatement)
    def translate(self, ctx):
        ectx = ctx.ectx()
        translate(self.expression, ectx)
        ectx.release_tempvars()

    @translate.on(ir.Return)
    def translate(self, ctx):
        ectx = ctx.ectx()
        if self.expression:
            rtype, tempvar = translate(self.expression, ectx)
        else:
            rtype = 'void'
        fctx = ctx.fctx
        expected_rtype = fctx.fdef.return_type
        if expected_rtype != rtype:
            raise Error([fctx.fdef.token, self.token],
                        f'Function was declared to return {expected_rtype} '
                        f'but tried to return {rtype} instead')
        if rtype == 'void':
            _release_for_return(ctx, ectx.src)
            ectx.src += 'return;'
        else:
            ectx.release_tempvars(tempvar)
            _release_for_return(ctx, ectx.src)
            ectx.src += f'return {tempvar};'

        return True

    @translate.on(ir.Block)
    def translate(self, pctx):
        pctx.src += '{'
        prologue = pctx.src.spawn(1)
        ctx = pctx.bctx(1)
        epilogue = pctx.src.spawn(1)
        pctx.src += '}'
        early_return = False
        for i, statement in enumerate(self.statements):
            if translate(statement, ctx):
                early_return = True
                if i + 1 < len(self.statements):
                    raise Error([self.statements[i + 1].token],
                                'Unreachable statement')
                break

        # Declare the local variables for C
        # To be C89 compatible, we need all variable definitions
        # to appear at the beginning of the block.
        for vdef in ctx.scope.local_definitions:
            assert isinstance(vdef, ir.VariableDefinition), vdef
            prologue += f'{_cproto(vdef, ctx)} = {ctx.czero(vdef.type)};'

        # If there's an early return, there's no need to have an
        # epilogue and generate unreachable code.
        if not early_return:
            # If we don't have an early return, we should
            # make sure to release all local variables defined in
            # this block before exiting
            # We should also release in LIFO order
            for vdef in reversed(ctx.scope.local_definitions):
                if vdef.type not in PRIMITIVE_TYPES:
                    epilogue += _defcrelease(vdef, ctx)

    @translate.on(ir.If)
    def translate(self, ctx):
        ctx.src += '{'
        ctx.src += 'KLC_bool b;'
        ectx = ctx.ectx()
        rtype, tempvar = translate(self.condition, ectx)
        ectx.src += f'b = {_ctruthy(ctx, rtype, tempvar)};'
        ectx.release_tempvars()
        ctx.src += 'if (b)'
        translate(self.body, ctx)
        if self.other is not None:
            ctx.src += 'else'
            translate(self.other, ctx)
        ctx.src += '}'

    @translate.on(ir.While)
    def translate(self, ctx):
        ctx.src += 'while (1) {'
        ctx.src += 'KLC_bool b;'
        ectx = ctx.ectx()
        rtype, tempvar = translate(self.condition, ectx)
        ectx.src += f'b = {_ctruthy(ctx, rtype, tempvar)};'
        ectx.release_tempvars()
        ctx.src += 'if (!b) { break; }'
        translate(self.body, ctx)
        ctx.src += '}'

    @translate.on(ir.Program)
    def translate(self):
        ctx = GlobalTranslationContext(self)
        for inc in self.env['@vars']['C_HEADERS']:
            ctx.inc += f'#include "{inc}"'
        for ptype, mnames in sorted(_primitive_method_names.items()):
            methodmap = _compute_method_map(
                token=self.token,
                cname=ptype,
                method_names=mnames,
                trait_names=['Object'],
                ctx=ctx)
            _write_ctypeinfo(ctx.src, ptype, methodmap, use_null_deleter=True)
        for d in self.definitions:
            translate(d, ctx)
        return str(ctx.out)

    @translate.on(ir.GlobalVariableDefinition)
    def translate(self, ctx):
        n = encode(self.name)
        ifn = encode(self.name + '#init')
        ctype = ctx.cdecltype(self.type)
        ctx.hdr += f'{ctype} KLC_get_global{n}();'
        ctx.src += f'int KLC_initialized_global{n} = 0;'
        ctx.src += f'{ctype} KLC_global{n} = {ctx.czero(self.type)};'

        ctx.src += f'{ctype} KLC_get_global{n}() ' '{'
        ctx.src += f'  if (!KLC_initialized_global{n}) ' '{'
        ctx.src += f'    KLC_global{n} = {ifn}();'
        ctx.src += f'    KLC_initialized_global{n} = 1;'
        if self.type in PRIMITIVE_TYPES:
            pass
        elif self.type == 'var':
            ctx.src += f'    KLC_release_var_on_exit(KLC_global{n});'
        else:
            ctx.src += f'    KLC_release_object_on_exit((KLC_header*) KLC_global{n});'
        ctx.src += '  }'
        src1 = ctx.src.spawn(1)
        src1 += _cretain(ctx, self.type, f'KLC_global{n}')
        src1 += f'return KLC_global{n};'
        ctx.src += '}'

    @translate.on(ir.FunctionDefinition)
    def translate(self, gctx: GlobalTranslationContext):
        ctx = gctx.fctx(self)

        rt = self.return_type
        if rt not in PRIMITIVE_TYPES and rt != 'var':
            rtnode = ctx.scope.get(self.return_type, [self.token])
            if isinstance(rtnode, ir.TraitDefinition):
                raise Error(
                    [vardef.token],
                    f'Declaring trait as return type not supported')
            if not isinstance(rtnode, ir.TypeDefinition):
                raise Error([vardef.token], f'{vardef.type} is not a type')

        _translate_untyped(self, ctx)

        ctx.hdr += _cproto(self, ctx) + ';'
        if self.body:
            ctx.src += _cproto(self, ctx)
            translate(self.body, ctx)

    @translate.on(ir.TraitDefinition)
    def translate(self, ctx: GlobalTranslationContext):
        # Verify that there's no circular trait inheritance,
        # and in the process, also verify all entries listed in 'traits'
        # are actually defined and are traits.
        _trait_method_map(self, ctx)

    @translate.on(ir.ClassDefinition)
    def translate(self, ctx: GlobalTranslationContext):
        _check_all_are_traits(self.token, self.traits, ctx)
        name = self.name
        n0 = encode0(name)
        cname = ctx.cname(name)
        cdecltype = ctx.cdecltype(name)

        del_name = _delname(n0)
        malloc_name = f'KLC_malloc{n0}'

        delete_proto = f'void {del_name}(KLC_header* robj, KLC_header** dq)'
        malloc_proto = f'{cdecltype} {malloc_name}()'

        ctx.hdr += delete_proto + ';'
        ctx.hdr += malloc_proto + ';'

        ctx.hdr += f'extern KLC_typeinfo KLC_type{n0};'

        _write_ctypeinfo(
            src=ctx.src,
            cname=name,
            methodmap=_xmethod_map(self, ctx),
            untyped_methods=self.untyped_methods)

        if self.extern:
            return

        _translate_field_implementations(self, ctx)

        # if extern, this typedef should already exist
        ctx.fwd += f'typedef struct {cname} {cname};'

        ctx.hdr += f'struct {cname} ' '{'
        ctx.hdr += '  KLC_header header;'
        for field in self.fields:
            ctx.hdr += f'  {_cproto(field, ctx)};'
        ctx.hdr += '};'

        ctx.src += delete_proto + ' {'
        objfields = [f for f in self.fields if f.type not in PRIMITIVE_TYPES]
        if objfields:
            ctx.src += f'  {cdecltype} obj = ({cdecltype}) robj;'
            for field in objfields:
                cfname = ctx.cname(field.name)
                if field.type == 'var':
                    ctx.src += f'  KLC_partial_release_var(obj->{cfname}, dq);'
                else:
                    ctx.src += f'  KLC_partial_release((KLC_header*) obj->{cfname}, dq);'
        ctx.src += '}'

        ctx.src += malloc_proto + ' {'
        ctx.src += f'  {cdecltype} obj = ({cdecltype}) malloc(sizeof({cname}));'
        ctx.src += f'  KLC_init_header(&obj->header, &KLC_type{n0});'
        for field in self.fields:
            cfname = ctx.cname(field.name)
            ctx.src += f'  obj->{cfname} = {ctx.czero(field.type)};'
        ctx.src += '  return obj;'
        ctx.src += '}'

    _cproto = Multimethod('_cproto')

    @_cproto.on(ir.BaseVariableDefinition)
    def _cproto(self, ctx):
        return f'{ctx.cdecltype(self.type)} {ctx.cname(self.name)}'

    @_cproto.on(ir.FunctionDefinition)
    def _cproto(self, ctx):
        crt = ctx.cdecltype(self.return_type)
        cname = ctx.cname(self.name)
        cparams = ', '.join(_cproto(p, ctx) for p in self.params)
        return f'{crt} {cname}({cparams})'

    def _cname(self: ir.BaseVariableDefinition, ctx):
        return ctx.cname(self.name)

    def _translate_field_implementations(self: ir.ClassDefinition, ctx):
        this_proto = f"{ctx.cdecltype(self.name)} {ctx.cname('this')}"
        for field in self.fields:
            field_ref = f'{encode("this")}->{ctx.cname(field.name)}'
            ctype = ctx.cdecltype(field.type)

            ## GETTER
            getter_name = f'{self.name}:GET{field.name}'
            getter_cname = ctx.cname(getter_name)
            getter_proto = f'{ctype} {getter_cname}({this_proto})'
            ctx.src += getter_proto + '{'
            sp = ctx.src.spawn(1)
            ctx.src += '}'
            sp += _cretain(ctx, field.type, f'({field_ref})')
            sp += f'return {field_ref};'

            # SETTER
            setter_name = f'{self.name}:SET{field.name}'
            setter_cname = ctx.cname(setter_name)
            setter_proto = f'{ctype} {setter_cname}({this_proto}, {ctype} v)'
            ctx.src += setter_proto + '{'
            sp = ctx.src.spawn(1)
            ctx.src += '}'
            # We have to retain 'v' twice:
            #  once for returning this value, and
            #  once more for attaching it to a field of this object.
            sp += _cretain(ctx, field.type, 'v')
            sp += _cretain(ctx, field.type, 'v')
            sp += _crelease(ctx, field.type, f'({field_ref})')
            sp += f'{field_ref} = v;'
            sp += 'return v;'

    def _defcretain(self: ir.BaseVariableDefinition, ctx):
        return _cretain(ctx, self.type, _cname(self, ctx))

    def _defcrelease(self: ir.BaseVariableDefinition, ctx):
        return _crelease(ctx, self.type, _cname(self, ctx))

    def _trait_method_map(self, ctx, stack=None):
        if self._trait_method_map is None:
            self._trait_method_map = _compute_method_map(
                token=self.token,
                cname=self.name,
                method_names=self.methods,
                trait_names=self.traits,
                ctx=ctx,
                stack=stack)

        return self._trait_method_map

    def _release_for_return(ctx, src):
        # Before returning, we should release all local variables
        # We need to take care not to release function parameters
        # or global variables
        scope = ctx.scope
        while scope:
            for vdef in reversed(scope.local_definitions):
                if isinstance(vdef, ir.VariableDefinition):
                    src += _defcrelease(vdef, ctx)
            scope = scope.parent

    def _compute_method_map(token, cname, method_names, trait_names, ctx, stack=None):
        ctx = ctx.gctx  # double check that this is the global context
        stack = [] if stack is None else stack
        if cname in stack:
            raise Error([tdef.token for tdef in stack],
                        f'Circular trait inheritance')
        _check_all_are_traits(token, trait_names, ctx)
        method_map = {mname: f'{cname}:{mname}' for mname in method_names}
        traits = [ctx.scope.get(n, [token]) for n in trait_names]
        stack.append(cname)
        # MRO is DFS
        for trait in traits:
            for mname, mfname in _trait_method_map(trait, ctx, stack).items():
                if mname not in method_map:
                    method_map[mname] = mfname
        stack.pop()
        return method_map

    def _xmethod_map(self, ctx):
        if self._method_map is None:
            self._method_map = _compute_method_map(
                token=self.token,
                cname=self.name,
                method_names=self.methods,
                trait_names=self.traits,
                ctx=ctx)
        return self._method_map

    def _delname(cname):
        return f'KLC_delete{cname}'

    def _crelease(ctx, type_, cname):
        if type_ in PRIMITIVE_TYPES:
            return ''
        elif type_ == 'var':
            return f'KLC_release_var({cname});'
        else:
            return f'KLC_release((KLC_header*) {cname});'

    def _cretain(ctx, type_, cname):
        if type_ in PRIMITIVE_TYPES:
            return ''
        elif type_ == 'var':
            return f'KLC_retain_var({cname});'
        else:
            return f'KLC_retain((KLC_header*) {cname});'

    def _ctruthy(ctx, type_, cname):
        if type_ == 'bool':
            return cname
        elif type_ in ('int', 'double'):
            return f'({cname} != 0)'
        elif type_ in ('function', 'type'):
            return '1'
        elif type_ == 'null':
            return '0'
        elif type_ == 'var':
            return f'KLC_truthy({cname})'
        else:
            return f'KLC_truthy(KLC_object_to_var((KLC_header*) {cname}))'

    def _write_ctypeinfo(src, cname, methodmap, use_null_deleter=False,
                         untyped_methods=None):
        # For primitive types, it's silly to have a deleter, so
        # use_null_deleter allows caller to control this

        cn = encode0(cname)

        del_name = _delname(cn)
        if methodmap or untyped_methods:
            # If there are 'extern' methods, add entries for them here
            if untyped_methods is not None:
                for mname in untyped_methods:
                    methodmap[mname] = f'{cname}:{mname}'
            src += f'static KLC_methodinfo KLC_methodarray{cn}[] = ' '{'
            for mname, mfname in sorted(methodmap.items()):
                src += '  {' f'"{mname}", KLC_untyped{encode(mfname)}' '},'
            src += '};'

        src += f'static KLC_methodlist KLC_methodlist{cn} = ' '{'
        src += f'  {len(methodmap)},'
        src += f'  KLC_methodarray{cn},' if methodmap else '  NULL,'
        src += '};'

        src += f'KLC_typeinfo KLC_type{cn} = ' '{'
        src += f'  "{cname}",'
        src += '  NULL,' if use_null_deleter else f'  &{del_name},'
        src += f'  &KLC_methodlist{cn},'
        src += '};'

    def _check_all_are_traits(token, traits, ctx):
        for trait_name in traits:
            trait_defn = ctx.scope.get(trait_name, [token])
            if not isinstance(trait_defn, ir.TraitDefinition):
                raise Error([token, trait_defn.token],
                            f'{trait_name} is not a trait')

    def _translate_fcall(ctx, token, defn, argtriples):
        argtempvars = []
        for param, (argtok, argtype, argtempvar) in zip(defn.params, argtriples):
            if argtype != 'void' and param.type == 'var':
                argtempvar = ctx.varify(argtype, argtempvar)
                argtype = 'var'
            if argtype != param.type:
                raise Error([param.token, argtok],
                            f'Expected {param.type} but got {argtype}')
            argtempvars.append(argtempvar)
        if len(defn.params) != len(argtriples):
            raise Error([token, defn.token],
                        f'{len(defn.params)} args expected '
                        f'but got {len(argtriples)}')
        argsstr = ', '.join(argtempvars)
        with with_frame(ctx, token):
            if defn.return_type == 'void':
                ctx.src += f'{ctx.cname(defn.name)}({argsstr});'
                return 'void', None
            else:
                tempvar = ctx.mktemp(defn.return_type)
                ctx.src += f'{tempvar} = {ctx.cname(defn.name)}({argsstr});'
                return defn.return_type, tempvar

    def _no_such_method_error(token, name, ownertype):
        return Error(
            [token],
            f'Method {name} does not exist for type {ownertype}')

    def _translate_mcall(ctx, token, name, argtriples):
        if len(argtriples) < 1:
            raise Error([token], 'FUBAR: mcall requires at least one arg')

        _, ownertype, ownertempvar = argtriples[0]

        if ownertype == 'void':
            raise Error([token], f'Cannot call method on void type')

        if ownertype == 'var':
            argtempvars = []
            for argtoken, argtype, argtempvar in argtriples:
                if argtype == 'void':
                    raise Error(
                        [arg.token],
                        'void expression cannot be used as an argument')
                argtempvar = ctx.varify(argtype, argtempvar)
                argtempvars.append(argtempvar)
            tarr = ctx.mktemparr(len(argtriples))
            tv = ctx.mktemp('var')
            for i, argtempvar in enumerate(argtempvars):
                ctx.src += f'{tarr}[{i}] = {argtempvar};'
            with with_frame(ctx, token):
                ctx.src += (
                    f'{tv} = KLC_mcall("{name}", '
                    f'{len(argtempvars)}, {tarr});'
                )
            return 'var', tv
        else:
            # Check that this method actually exists on this type
            if ownertype in PRIMITIVE_TYPES:
                if name not in _primitive_method_names[ownertype]:
                    raise _no_such_method_error(token, name, ownertype)
                fname = f'{ownertype}:{name}'
            else:
                cdef = ctx.scope.get(ownertype, [token])
                assert isinstance(cdef, ir.ClassDefinition), cdef
                if name not in _xmethod_map(cdef, ctx):
                    raise _no_such_method_error(token, name, ownertype)
                fname = _xmethod_map(cdef, ctx)[name]

            # TODO: Consider looking up from global context
            # to avoid coincidental names that shadow method names
            defn = ctx.scope.get(fname, [token])
            if not isinstance(defn, ir.FunctionDefinition):
                raise Error([token], f'FUBAR: shadowed method {fname}')

            return _translate_fcall(ctx, token, defn, argtriples)

    def _untyped_cproto(self, ctx):
        return f'KLC_var KLC_untyped{encode(self.name)}(int argc, const KLC_var* argv)'

    def _translate_untyped(self, ctx):
        name = self.name
        ctx.hdr += _untyped_cproto(self, ctx) + ';'
        ctx.hdr += f'KLC_functioninfo KLC_functioninfo{encode(name)} = ' '{'
        ctx.hdr += f'  "{name}",'
        ctx.hdr += f'  KLC_untyped{encode(self.name)},'
        ctx.hdr += '};'
        ctx.src += _untyped_cproto(self, ctx) + '{'
        src = ctx.src.spawn(1)
        ctx.src += '}'
        for i, param in enumerate(self.params):
            src += f'{ctx.cdecltype(param.type)} arg{i};'
        src += f'if (argc != {len(self.params)}) ' '{'
        src += f'  KLC_errorf("Function \'%s\' expected %d args but got %d", \"{self.name}\", {len(self.params)}, argc);'
        src += '}'
        for i, param in enumerate(self.params):
            src += f'arg{i} = {ctx.unvarify(param.type, f"argv[{i}]")};'
        argsstr = ', '.join(f'arg{i}' for i in range(len(self.params)))
        call = f'{ctx.cname(self.name)}({argsstr})'
        if self.return_type == 'void':
            src += f'{call};'
            src += 'return KLC_null;'
        else:
            src += f'return {ctx.varify(self.return_type, call)};'

    def _translate_value(self: ir.Expression, ctx):
        etype, evar = translate(self, ctx)
        if etype == 'void':
            raise Error([self.token], 'void type not allowed here')
        return etype, evar

    @contextlib.contextmanager
    def with_frame(ctx, token):
        src = ctx.src
        filename = token.source.filename.replace(os.sep, '/')
        fname = ctx.fctx.fdef.name
        src += f'KLC_push_frame("{filename}", "{fname}", {token.lineno});'
        yield
        src += f'KLC_pop_frame();'


@Namespace
def parser(ns):

    class MacroScope:
        def __init__(self, parent, table=None, stack=None):
            if parent is None:
                if stack is None:
                    stack = []
            else:
                if stack is None:
                    stack = parent.stack
                else:
                    raise TypeError((parent, stack))
            self.parent = parent
            self.table = dict() if table is None else table
            self.stack = stack

        @contextlib.contextmanager
        def using(self, token):
            self.stack.append(token)
            yield
            self.stack.pop()

        def __getitem__(self, key):
            if key in self.table:
                return self.table[key]
            elif self.parent is not None:
                return self.parent[key]
            else:
                raise Error(self.stack, f'No such macro name {repr(key)}')

        def __setitem__(self, key, value):
            if key in self.table:
                self.table[key] = value
            elif self.parent is not None:
                self.parent[key] = value
            else:
                raise Error(self.stack, f'No such macro name {repr(key)}')

        def define(self, key, value):
            if key in self.table:
                raise Error(
                    self.stack, f'Macro name already defined {repr(key)}')
            self.table[key] = value

    class MacroLambda(typing.NamedTuple):
        token: Token
        scope: MacroScope
        params: typing.List[str]
        body: list

    class Context(typing.NamedTuple):
        defs: list
        macro_scope: MacroScope
        inside_try: bool

        def copy(self, **kwargs):
            for key in Context.__annotations__.keys():
                if key not in kwargs:
                    kwargs[key] = getattr(self, key)
            return Context(**kwargs)

    @ns
    def parse(source, local_prefix, env):
        plat = sys.platform

        # NOTE: This is really for documentation purposes, so that
        # plat is not something unexpected
        assert plat in (
            'darwin',  # OS X
            'win32',
            'linux',
        ), plat
        if env is None:
            env = {
                '@stack': [],
                '@intgen': [0],
                '@ntempvar': [0],
                '@cache': dict(),
                '@vars': {
                    'PLATFORM': plat,
                },
            }
            env['@macro_scope'] = MacroScope(MacroScope(None, env['@vars']))
        main_node = parse_one_source(source, local_prefix, env)
        defs = list(main_node.definitions)
        for lib_node in env['@cache'].values():
            defs.extend(lib_node.definitions)
        return ir.Program(main_node.token, defs, env)

    def _has_upper(s):
        return s.lower() != s

    def _is_special_method_name(name):
        return (name in _special_method_names or
                any(name.startswith(p) for p in _special_method_prefixes))

    def _check_method_name(token, name):
        # TODO: Consider removing this restriction
        if _has_upper(name[0]) and not _is_special_method_name(name):
            raise Error(
                [token],
                f'Only special methods may start with an upper case letter')

    def peek_local_symbols(tokens):
        """Figure out what classes, traits, functions and variables
        are declared in this file.
        The parser needs to know this information ahead of time
        in order to be able to properly qualify names.
        """
        i = 0

        def peek(j=0):
            return tokens[min(i + j, len(tokens) - 1)]

        def at(type, j=0):
            return peek(j).type == type

        def at_seq(types, j=0):
            return all(at(t, i) for i, t in enumerate(types, j))

        def gettok():
            nonlocal i
            token = peek()
            i += 1
            return token

        def at_function():
            return at_seq(['NAME', 'NAME', '('])

        def at_class():
            return at('class')

        def at_trait():
            return at('trait')

        def at_vardef():
            return (
                at_seq(['NAME', 'NAME', '=']) or
                at_seq(['NAME', 'NAME', '\n']))

        def skip_to_matching():
            op = peek().type
            # if op is not one of these, there is a bug in this file,
            # this function must be called at one of these places
            cl = {
                '(': ')',
                '[': ']',
                '{': '}',
            }[op]
            depth = 1
            gettok()
            while depth:
                t = gettok().type
                if t == op:
                    depth += 1
                elif t == cl:
                    depth -= 1

        def expect(type):
            if not at(type):
                raise Error([peek()], f'Expected {type} but got {peek()}')
            return gettok()

        symbols = set()

        while not at('EOF'):
            if at_function() or at_vardef():
                expect('NAME')
                symbols.add(expect('NAME').value)
            elif at_class() or at_trait():
                gettok()
                symbols.add(expect('NAME').value)
            elif at('(') or at('[') or at('{'):
                skip_to_matching()
            else:
                gettok()

        return symbols

    def update_tokens(tokens, i, module_map):
        """
        Creates a new list of tokens to take care of issues
        with namespacing.

        The first i tokens are copied as is. It's assumed that
        the first i tokens are 'import' statements, so there's
        no need to touch these.

        whenever we sees a module alias, we will
        expect to see a '.name', and resolve that full
        name into a single NAME token.
        """
        new_tokens = tokens[:i]
        while i < len(tokens):
            token = tokens[i]
            if (token.type == '.' and
                    i + 1 < len(tokens) and
                    tokens[i + 1].type == 'NAME'):
                # If dot is followed by a NAME, we probably don't
                # want to update that NAME
                new_tokens.append(token)
                new_tokens.append(tokens[i + 1])
                i += 2
                continue
            if token.type == 'NAME' and token.value in module_map:
                module_name = module_map[token.value]
                if (i + 2 < len(tokens) and
                        tokens[i + 1].type == '.' and
                        tokens[i + 2].type == 'NAME'):
                    new_name = f'{module_name}.{tokens[i + 2].value}'
                    i += 3
                    new_tokens.append(Token(
                        type='NAME',
                        value=new_name,
                        source=token.source,
                        i=token.i))
                    continue
                raise Error(
                    [token],
                    'Module aliases must be followed by a dot and NAME')
            # In most cases, we just want to take that token as is
            i += 1
            new_tokens.append(token)
        return new_tokens

    def parse_one_source(source, local_prefix, env):
        tokens = lexer.lex(source)
        i = 0
        indent_stack = []
        cache = env['@cache']
        stack = env['@stack']
        intgen = env['@intgen']
        module_map = dict()
        local_symbols = (
            set() if local_prefix is None else peek_local_symbols(tokens)
        )

        def mktempvar():
            i = env['@ntempvar'][0]
            env['@ntempvar'][0] += 1
            return f'#tempvar#{i}'

        def peek(j=0):
            nonlocal i
            if should_skip_newlines():
                while i < len(tokens) and tokens[i].type == '\n':
                    i += 1
            return tokens[min(i + j, len(tokens) - 1)]

        def at(type, j=0):
            return peek(j).type == type

        def should_skip_newlines():
            return indent_stack and indent_stack[-1]

        @contextlib.contextmanager
        def skipping_newlines(skipping):
            indent_stack.append(skipping)
            yield
            indent_stack.pop()

        def gettok():
            nonlocal i
            token = peek()
            i += 1
            return token

        def consume(type):
            if at(type):
                return gettok()

        def expect(type):
            if not at(type):
                raise Error([peek()], f'Expected {type} but got {peek()}')
            return gettok()

        def expect_name(name):
            if not at_name(name):
                token = expect('NAME')
                raise Error([token], f'Expected name {name} but got {token.value}')
            return expect('NAME')

        def at_name(name, j=0):
            return at('NAME', j) and peek(j).value == name

        def expect_delim():
            token = peek()
            if not consume(';'):
                expect('\n')
            while consume(';') or consume('\n'):
                pass
            return token

        def consume_delim():
            if at_delim():
                return expect_delim()

        def at_delim(j=0):
            return at(';', j) or at('\n', j)

        def replace_tokens_with_update():
            # TODO: *Sigh* this is a hack.
            nonlocal tokens
            tokens = update_tokens(tokens, i, module_map)

        def parse_program():
            consume_delim()
            token = peek()
            defs = []
            ctx = Context(
                defs=defs,
                macro_scope=env['@macro_scope'],
                inside_try=False)
            while at('import'):
                parse_import(ctx)
            replace_tokens_with_update()
            consume_delim()
            while not at('EOF'):
                parse_global_definition(ctx)
                consume_delim()
            return ir.Program(token, defs, env)

        def parse_global_definition(ctx):
            if at('#'):
                parse_macro(ctx)
                return

            if at('trait'):
                parse_trait_definition(ctx)
                return

            if at('class') or at('extern') and at('class', 1):
                parse_class_definition(ctx)
                return

            if (at('NAME') and at('NAME', 1) and at('(', 2) or
                    at('try') and at('NAME', 1) and at('NAME', 2) and
                        at('(', 3)):
                parse_function_definition(ctx)
                return

            if (at('extern') or at('NAME')) and at('NAME', 1):
                parse_global_variable_definition(ctx)
                return

            raise Error([peek()], f'Expected class, function or variable definition')

        def parse_import(ctx):
            token = expect('import')
            parts = [expect('NAME').value]
            while consume('.'):
                parts.append(expect('NAME').value)
            alias = expect('NAME').value if consume('as') else parts[-1]
            expect_delim()
            name = '.'.join(parts)
            module_map[alias] = name
            upath = os.path.abspath(os.path.realpath(
                os.path.join(_scriptdir, 'lib', name.replace('.', os.sep) + '.k')
            ))

            if upath in cache:
                return

            if upath in [p for p, _ in stack]:
                toks = [t for _, t in stack]
                raise Error(toks + [token], 'import cycle')

            if not os.path.isfile(upath):
                raise Error([token], f'File {upath} ({upath}) does not exist')

            with open(upath) as f:
                data = f.read()

            try:
                stack.append((upath, token))
                cache[upath] = parse_one_source(Source(upath, data), name, env)
            finally:
                stack.pop()

        def parse_macro(ctx):
            token = expect('#')
            if at('('):
                parse_macro_expression()(ctx.macro_scope)
            else:
                raise Error([peek()], f'Unrecognized macro type')

        def _check_args(token, fn, expect, args):
            if len(args) != expect:
                raise Error(
                    [token],
                    f'Macro function {fn} expects {expect} args '
                    f'but got {len(args)}')

        def parse_macro_name():
            return (
                gettok().type if peek().type in lexer.KEYWORDS else
                expect('NAME').value
            )

        def parse_macro_expression():
            # The result of parsing a macro expression is just a Python
            # function that accepts no arguments. Simply running the
            # function will cause the macro expression to be evaluated.
            # Result of macro expressions must be one of the following:
            #  * number (double)
            #  * string
            #  * list
            token = peek()
            if consume('('):
                with skipping_newlines(True):
                    fn = parse_macro_name()
                    if fn in ('global', 'define', 'set'):
                        vname = parse_macro_name()
                        expr = parse_macro_expression()
                        expect(')')
                        if fn == 'global':
                            def run(mscope, token=token):
                                with mscope.using(token):
                                    result = env['@vars'][vname] = expr(mscope)
                                    return result
                        elif fn == 'define':
                            def run(mscope, token=token):
                                with mscope.using(token):
                                    result = expr(mscope)
                                    mscope.define(vname, result)
                                    return result
                        elif fn == 'set':
                            def run(mscope, token=token):
                                with mscope.using(token):
                                    result = mscope[vname] = expr(mscope)
                                    return result
                        else:
                            raise Error([token], 'FUBAR')
                        return run
                    elif fn == 'lambda':
                        expect('(')
                        params = []
                        while not consume(')'):
                            params.append(parse_macro_name())
                        bodyexprs = []
                        while not consume(')'):
                            bodyexprs.append(parse_macro_expression())
                        def run(mscope):
                            return MacroLambda(
                                token, mscope, params, bodyexprs)
                        return run
                    argexprs = []
                    while not consume(')'):
                        argexprs.append(parse_macro_expression())
                    def run(mscope, token=token):
                        if fn == 'if':
                            i = 0
                            while i + 1 < len(argexprs):
                                if argexprs[i](mscope):
                                    return argexprs[i + 1](mscope)
                                i += 2
                            return argexprs[i](mscope) if i < len(argexprs) else 0.0
                        elif fn == 'begin':
                            last = 0.0
                            for expr in argexprs:
                                last = expr(mscope)
                            return last
                        args = [expr(mscope) for expr in argexprs]
                        if fn == 'eq':
                            _check_args(token, fn, 2, args)
                            return args[0] == args[1]
                        elif fn == 'str':
                            return ''.join(map(str, args))
                        elif fn == 'append':
                            _check_args(token, fn, 2, args)
                            args[0].append(args[1])
                            return 0.0
                        elif fn == 'print':
                            # Should be used for debugging purposes only!!
                            _check_args(token, fn, 1, args)
                            print(args[0])
                        elif fn == 'error':
                            _check_args(token, fn, 1, args)
                            with mscope.using(token):
                                raise Error(mscope.stack, str(args[0]))
                        elif fn == 'add':
                            if len(args) < 1:
                                raise Error([token], 'add expects at least 1 arg')
                            result = args[0]
                            for arg in args[1:]:
                                result += arg
                            return result
                        elif fn == 'subtract':
                            _check_args(token, fn, 2, args)
                            return args[0] - args[1]
                        elif fn == 'multiply':
                            _check_args(token, fn, 2, args)
                            return args[0] * args[1]
                        elif fn == 'divide':
                            _check_args(token, fn, 2, args)
                            return args[0] / args[1]
                        elif fn == 'modulo':
                            _check_args(token, fn, 2, args)
                            return args[0] % args[1]
                        elif fn == 'lt':
                            _check_args(token, fn, 2, args)
                            return args[0] < args[1]
                        else:
                            # TODO: Stop this hack and support
                            # expressions of the form:
                            #  ((make_function) arguments_to_new_function)
                            with mscope.using(token):
                                lambda_ = mscope[fn]
                                new_scope = MacroScope(lambda_.scope)
                                if len(lambda_.params) != len(args):
                                    with mscope.using(lambda_.token):
                                        raise Error(
                                            mscope.stack,
                                            f'Expected {len(lambda_.params)} '
                                            f'args but got {len(args)}')
                                for param, arg in zip(lambda_.params, args):
                                    new_scope.define(param, arg)
                                for expr in lambda_.body:
                                    last = expr(new_scope)
                                return last
                    return run
            elif consume('['):
                with skipping_newlines(True):
                    exprs = []
                    while not consume(']'):
                        exprs.append(parse_macro_expression())
                    def run(mscope):
                        return [expr(mscope) for expr in exprs]
                    return run
            elif at('INT') or at('FLOAT'):
                value = float(gettok().value)
                return lambda mscope: value
            elif at('STRING'):
                value = expect('STRING').value
                return lambda mscope: value
            elif at('NAME'):
                name = parse_macro_name()
                def run(mscope, token=token):
                    with mscope.using(token):
                        return mscope[name]
                return run
            else:
                raise Error(
                    [token],
                    f'Expected macro expression but got {repr(token.type)}')

        def expect_maybe_exported_name():
            name = expect('NAME').value
            if name in local_symbols:
                name = f'{local_prefix}.{name}'
            return name

        def expect_type():
            return expect_maybe_exported_name()

        def parse_global_variable_definition(ctx):
            token = peek()
            extern = bool(consume('extern'))
            vtype = expect_type()
            vname = expect_exported_name()
            ctx.defs.append(ir.GlobalVariableDefinition(token, extern, vtype, vname))
            ifname = f'{vname}#init'
            if extern:
                initf = ir.FunctionDefinition(
                    token,
                    vtype,
                    ifname,
                    [],
                    None)
            else:
                if consume('='):
                    expr = parse_expression(ctx)
                    initf = ir.FunctionDefinition(
                        token,
                        vtype,
                        ifname,
                        [],
                        ir.Block(token, [ir.Return(token, expr)]))
                else:
                    initf = ir.FunctionDefinition(
                        token,
                        vtype,
                        ifname,
                        [],
                        ir.Block(token, [
                            ir.VariableDefinition(token, True, vtype, 'ret', None),
                            ir.Return(token, ir.Name(token, 'ret')),
                        ]))
            ctx.defs.append(initf)
            expect_delim()

        def parse_function_definition(ctx):
            token = peek()
            try_ = consume('try')
            return_type = expect_type()
            name = expect_exported_name()
            params = parse_params()
            if at('{'):
                body = parse_block(ctx.copy(inside_try=try_))
            else:
                body = None
                expect_delim()
            if try_:
                tryname = f'{name}%try'
                ctx.defs.append(ir.FunctionDefinition(
                    token, return_type, name, params,
                    ir.Block(token, [
                        ir.Return(token,
                            ir.Cast(token,
                                ir.MethodCall(token,
                                    ir.FunctionCall(token, tryname, [
                                        ir.Name(token, p.name)
                                        for p in params
                                    ]),
                                    'orDie', []
                                ),
                                return_type,
                            ),
                        ),
                    ])))
                ctx.defs.append(ir.FunctionDefinition(
                    token, 'Try', tryname, params, body))
            else:
                ctx.defs.append(ir.FunctionDefinition(
                    token, return_type, name, params, body))

        def expect_non_exported_name():
            token = peek()
            name = expect('NAME').value
            if local_prefix is not None and name in local_symbols:
                raise Error([token], f'Global name {name} cannot be used here')
            return name

        def expect_exported_name():
            name = expect('NAME').value
            if local_prefix is None:
                return name
            assert name in local_symbols, (name, local_symbols)
            return f'{local_prefix}.{name}'

        def parse_class_definition(ctx):
            token = peek()
            extern = bool(consume('extern'))
            expect('class')
            name = expect_exported_name()
            traits = parse_trait_list()
            method_to_token_table = dict()
            member_to_token_table = dict()

            def mark_member(name, token):
                if name in member_to_token_table:
                    raise Error([token], f'Duplicate member definition {name}')
                member_to_token_table[name] = token

            def mark_method(name, token):
                mark_member(name, token)
                method_to_token_table[name] = token

            fields = None if extern else []
            untyped_methods = []
            newdef = None
            expect('{')
            consume_delim()
            while not consume('}'):
                if at('new'):
                    if newdef is not None:
                        raise Error(
                            [newdef.token, peek()],
                            'Only one constructor definition is allowed for '
                            'a class')
                    fname = f'{name}#new'
                    mtoken = expect('new')
                    declparams = parse_params()
                    if extern:
                        # For extern types, the 'new' function should return
                        # the constructed object. Further, the new function
                        # itself must be extern
                        if name == 'Closure':
                            # It sucks to special case so much for this one
                            # type, but I want to make sure that an actual
                            # expression of type 'Closure' is impossible.
                            # So for just this one, I want the return type
                            # to be var
                            rt = 'var'
                        else:
                            rt = name
                        body = None
                        expect_delim()
                        params = declparams
                    else:
                        # For normal types, the 'new' function initializes
                        # an already allocated object.
                        # TODO: Make 'new' return the actual object
                        # like extern types.
                        rt = 'void'
                        body = parse_block(ctx)
                        params = [ir.Parameter(mtoken, name, 'this')] + declparams
                    newdef = ir.FunctionDefinition(mtoken, rt, fname, params, body)
                    ctx.defs.append(newdef)
                elif at('NAME') and at('NAME', 1) and at_delim(2):
                    if extern:
                        raise Error(
                            [peek()],
                            'Extern classes cannot declare fields '
                            f'(in definition of class {name})')
                    ftoken = peek()
                    ftype = expect_type()
                    fname = expect('NAME').value
                    mark_member(fname, ftoken)
                    expect_delim()
                    fields.append(ir.Field(ftoken, ftype, fname))

                    # GET and SET methods are implemented specially
                    # during the class translation
                    getter_name = f'{name}:GET{fname}'
                    setter_name = f'{name}:SET{fname}'
                    ctx.defs.append(ir.FunctionDefinition(
                        ftoken,
                        ftype,
                        getter_name,
                        [ir.Parameter(ftoken, name, 'this')],
                        None))
                    ctx.defs.append(ir.FunctionDefinition(
                        ftoken,
                        ftype,
                        setter_name,
                        [ir.Parameter(ftoken, name, 'this'),
                         ir.Parameter(ftoken, ftype, 'value')],
                        None))
                    mark_method(f'GET{fname}', ftoken)
                    mark_method(f'SET{fname}', ftoken)
                elif at('extern'):
                    uftoken = expect('extern')
                    ufname = expect('NAME').value
                    expect_delim()
                    untyped_methods.append(ufname)
                    mark_member(ufname, uftoken)
                else:
                    mtoken = peek()
                    try_ = consume('try')
                    rtype = expect_type()
                    mname = expect('NAME').value
                    _check_method_name(mtoken, mname)
                    params = [ir.Parameter(mtoken, name, 'this')] + parse_params()
                    body = None if consume_delim() else parse_block(
                        ctx.copy(inside_try=try_))

                    # A method is mapped to a function with a special name,
                    # and an implicit first parameter.
                    fname = f'{name}:{mname}'

                    mark_method(mname, token)

                    if try_:
                        trymname = f'{mname}%try'
                        tryfname = f'{fname}%try'
                        assert tryfname == f'{name}:{trymname}'
                        mark_method(trymname, token)
                        ctx.defs.append(ir.FunctionDefinition(
                            token, rtype, fname, params,
                            ir.Block(token, [
                                ir.Return(token,
                                    ir.Cast(token,
                                        ir.MethodCall(token,
                                            ir.FunctionCall(token, tryfname, [
                                                ir.Name(token, p.name)
                                                for p in params
                                            ]),
                                            'orDie', []
                                        ),
                                        rtype,
                                    ),
                                ),
                            ])))
                        ctx.defs.append(ir.FunctionDefinition(
                            token, 'Try', tryfname, params, body))
                    else:
                        ctx.defs.append(ir.FunctionDefinition(
                            mtoken, rtype, fname, params, body))
                consume_delim()

            consume_delim()

            if not extern and newdef is None:
                ctx.defs.append(ir.FunctionDefinition(
                    token,
                    'void',
                    f'{name}#new',
                    [ir.Parameter(token, name, 'this')],
                    ir.Block(token, [])))

            method_names = sorted(method_to_token_table)
            ctx.defs.append(ir.ClassDefinition(
                token, name, traits, fields, method_names, untyped_methods))

        def parse_trait_definition(ctx):
            token = expect('trait')
            name = expect_exported_name()
            method_to_token_table = dict()
            traits = parse_trait_list()
            expect('{')
            consume_delim()
            while not consume('}'):
                mtoken = peek()
                rtype = expect_type()
                mname = expect('NAME').value
                params = parse_params()
                body = parse_block(ctx)
                _check_method_name(mtoken, mname)

                # A method is mapped to a function with a special name,
                # and an implicit first parameter.
                fname = f'{name}:{mname}'
                params = [ir.Parameter(mtoken, 'var', 'this')] + params

                if mname in method_to_token_table:
                    raise Error([method_to_token_table[mname], mtoken],
                                f'Duplicate method {name}.{mname}')

                method_to_token_table[mname] = mtoken

                ctx.defs.append(ir.FunctionDefinition(mtoken, rtype, fname, params, body))
                consume_delim()
            consume_delim()

            method_names = sorted(method_to_token_table)
            ctx.defs.append(ir.TraitDefinition(token, name, traits, method_names))

        def parse_trait_list():
            if consume('('):
                traits = []
                while not consume(')'):
                    traits.append(expect_type())
                    if not consume(','):
                        expect(')')
                        break
            else:
                traits = ['Object']
            return traits

        def parse_block(ctx):
            token = expect('{')
            with skipping_newlines(False):
                consume_delim()
                statements = []
                while not consume('}'):
                    statements.append(parse_statement(ctx))
                    consume_delim()
                return ir.Block(token, statements)

        def parse_statement(ctx):
            token = peek()

            if at('{'):
                return parse_block(ctx)

            if at_variable_definition():
                return parse_variable_definition(ctx)

            if consume('while'):
                expect('(')
                with skipping_newlines(True):
                    condition = parse_expression(ctx)
                    expect(')')
                body = parse_block(ctx)
                return ir.While(token, condition, body)

            if consume('for'):
                if consume('('):
                    init = []
                    if consume(';'):
                        pass
                    elif at_variable_definition():
                        init.append(parse_variable_definition(ctx))
                    else:
                        init.append(ir.ExpressionStatement(token, parse_expression(ctx)))
                        expect(';')
                    cond = ir.BoolLiteral(token, True) if at(';') else parse_expression(ctx)
                    expect(';')
                    incr = ir.ExpressionStatement(token, parse_expression(ctx))
                    expect(')')
                    raw_body = parse_block(ctx)
                    body = ir.Block(token, raw_body.statements + [incr])
                    return ir.Block(token, init + [
                        ir.While(token, cond, body),
                    ])
                else:
                    if at('NAME') and at('in', 1):
                        vtype = 'var'
                    else:
                        vtype = expect_type()
                    loopvar = expect_non_exported_name()
                    expect('in')
                    container_expr = parse_expression(ctx)
                    body = parse_block(ctx)
                    tempvar = mktempvar()
                    return ir.Block(token, [
                        ir.VariableDefinition(
                            token,
                            True,
                            None,
                            tempvar,
                            ir.MethodCall(token, container_expr, 'Iterator', [])),
                        ir.While(
                            token,
                            ir.MethodCall(token, ir.Name(token, tempvar), 'HasNext', []),
                            ir.Block(token, [
                                ir.VariableDefinition(
                                    token,
                                    True,
                                    None,
                                    loopvar,
                                    ir.Cast(
                                        token,
                                        ir.MethodCall(
                                            token,
                                            ir.Name(token, tempvar),
                                            'Next',
                                            []),
                                        vtype
                                    ),
                                ),
                                body,
                            ])
                        ),
                    ])

            if consume('with'):
                expr = parse_expression(ctx)
                exprtempvar = mktempvar()
                if consume('as'):
                    name = expect('NAME').value
                else:
                    name = mktempvar()
                body = parse_block(ctx)
                return ir.Block(token, [
                    ir.VariableDefinition(
                        token,
                        True,
                        None,
                        exprtempvar,
                        expr),
                    ir.VariableDefinition(
                        token,
                        True,
                        None,
                        name,
                        ir.MethodCall(token, ir.Name(token, exprtempvar), 'Enter', [])),
                    ir.VariableDefinition(
                        token,
                        True,
                        None,
                        mktempvar(),
                        ir.FunctionCall(token, '%With', [ir.Name(token, exprtempvar)])),
                    body,
                ])

            if consume('if'):
                expect('(')
                with skipping_newlines(True):
                    condition = parse_expression(ctx)
                    expect(')')
                body = parse_block(ctx)
                if consume('else'):
                    other = parse_statement(ctx)
                else:
                    other = None
                return ir.If(token, condition, body, other)

            if consume('return'):
                expression = None if at_delim() else parse_expression(ctx)
                expect_delim()
                if ctx.inside_try:
                    expression = ir.FunctionCall(
                        token, '%Success', [expression])
                return ir.Return(token, expression)

            if consume('raise'):
                if not ctx.inside_try:
                    raise Error([token], 'Raise outside a try function')
                expression = ir.FunctionCall(
                    token, '%Failure', [parse_expression(ctx)])
                return ir.Return(token, expression)

            expression = parse_expression(ctx)
            expect_delim()
            return ir.ExpressionStatement(token, expression)

        def parse_expression(ctx):
            return parse_conditional(ctx)

        def parse_conditional(ctx):
            expr = parse_logical_or(ctx)
            token = peek()
            if consume('?'):
                left = parse_expression(ctx)
                expect(':')
                right = parse_conditional(ctx)
                return ir.Conditional(token, expr, left, right)
            return expr

        def parse_logical_or(ctx):
            expr = parse_logical_and(ctx)
            while True:
                token = peek()
                if consume('or'):
                    right = parse_logical_and(ctx)
                    expr = ir.LogicalOr(token, expr, right)
                else:
                    break
            return expr

        def parse_logical_and(ctx):
            expr = parse_relational(ctx)
            while True:
                token = peek()
                if consume('and'):
                    right = parse_relational(ctx)
                    expr = ir.LogicalAnd(token, expr, right)
                else:
                    break
            return expr

        def parse_relational(ctx):
            expr = parse_bitwise_or(ctx)
            while True:
                token = peek()
                if consume('=='):
                    expr = ir.Equals(token, expr, parse_bitwise_or(ctx))
                elif consume('!='):
                    expr = ir.NotEquals(token, expr, parse_bitwise_or(ctx))
                elif consume('<'):
                    expr = ir.LessThan(token, expr, parse_bitwise_or(ctx))
                elif consume('<='):
                    expr = ir.LessThanOrEqual(token, expr, parse_bitwise_or(ctx))
                elif consume('>'):
                    expr = ir.GreaterThan(token, expr, parse_bitwise_or(ctx))
                elif consume('>='):
                    expr = ir.GreaterThanOrEqual(token, expr, parse_bitwise_or(ctx))
                elif consume('is'):
                    if consume('not'):
                        expr = ir.IsNot(token, expr, parse_bitwise_or(ctx))
                    else:
                        expr = ir.Is(token, expr, parse_bitwise_or(ctx))
                elif consume('in'):
                    expr = ir.In(token, expr, parse_bitwise_or(ctx))
                elif consume('not'):
                    expect('in')
                    expr = ir.LogicalNot(token, ir.In(token, expr, parse_bitwise_or(ctx)))
                else:
                    break
            return expr

        def parse_bitwise_or(ctx):
            expr = parse_bitwise_xor(ctx)
            while True:
                token = peek()
                if consume('|'):
                    expr = ir.MethodCall(token, expr, 'Or', [parse_bitwise_xor(ctx)])
                else:
                    break
            return expr

        def parse_bitwise_xor(ctx):
            expr = parse_bitwise_and(ctx)
            while True:
                token = peek()
                if consume('^'):
                    expr = ir.MethodCall(token, expr, 'Xor', [parse_bitwise_and(ctx)])
                else:
                    break
            return expr

        def parse_bitwise_and(ctx):
            expr = parse_bitwise_shift(ctx)
            while True:
                token = peek()
                if consume('&'):
                    expr = ir.MethodCall(token, expr, 'And', [parse_bitwise_shift(ctx)])
                else:
                    break
            return expr

        def parse_bitwise_shift(ctx):
            expr = parse_additive(ctx)
            while True:
                token = peek()
                if consume('>>'):
                    expr = ir.MethodCall(token, expr, 'Rshift', [parse_additive(ctx)])
                elif consume('<<'):
                    expr = ir.MethodCall(token, expr, 'Lshift', [parse_additive(ctx)])
                else:
                    break
            return expr

        def parse_additive(ctx):
            expr = parse_multiplicative(ctx)
            while True:
                token = peek()
                if consume('+'):
                    expr = ir.MethodCall(token, expr, 'Add', [parse_multiplicative(ctx)])
                elif consume('-'):
                    expr = ir.MethodCall(token, expr, 'Sub', [parse_multiplicative(ctx)])
                else:
                    break
            return expr

        def parse_multiplicative(ctx):
            expr = parse_unary(ctx)
            while True:
                token = peek()
                if consume('*'):
                    expr = ir.MethodCall(token, expr, 'Mul', [parse_unary(ctx)])
                elif consume('/'):
                    expr = ir.MethodCall(token, expr, 'Div', [parse_unary(ctx)])
                elif consume('%'):
                    expr = ir.MethodCall(token, expr, 'Mod', [parse_unary(ctx)])
                else:
                    break
            return expr

        def parse_unary(ctx):
            token = peek()
            if consume('-'):
                expr = parse_unary(ctx)
                if isinstance(expr, (ir.IntLiteral, ir.DoubleLiteral)):
                    type_ = type(expr)
                    return type_(expr.token, -expr.value)
                else:
                    return ir.MethodCall(token, expr, 'Neg', [])
            if consume('~'):
                expr = parse_unary(ctx)
                return ir.MethodCall(token, expr, 'Invert', [])
            if consume('!'):
                expr = parse_unary(ctx)
                return ir.LogicalNot(token, expr)
            return parse_pow(ctx)

        def parse_pow(ctx):
            expr = parse_postfix(ctx)
            token = peek()
            if consume('**'):
                expr = ir.MethodCall(token, expr, 'Pow', [parse_pow(ctx)])
            return expr

        def parse_postfix(ctx):
            expr = parse_primary(ctx)
            while True:
                token = peek()
                if consume('.'):
                    if consume('('):
                        cast_type = expect_type()
                        expect(')')
                        expr = ir.Cast(token, expr, cast_type)
                        continue
                    else:
                        try_ = consume('try')
                        if try_:
                            expect('.')
                        name = expect('NAME').value
                        if try_:
                            name = f'{name}%try'
                        if at('('):
                            args = parse_args(ctx)
                            expr = ir.MethodCall(token, expr, name, args)
                            continue
                        elif consume('='):
                            val = parse_expression(ctx)
                            expr = ir.MethodCall(token, expr, f'SET{name}', [val])
                            continue
                        else:
                            expr = ir.MethodCall(token, expr, f'GET{name}', [])
                            continue
                elif consume('['):
                    # x[i,...]       GetItem
                    # x[i,...] = v   SetItem
                    # x[a:b]         Slice
                    # x[:b]          SliceRight
                    # x[a:b]         SliceLeft
                    # x[a:b] = v     SetSlice
                    # x[:b] = v      SetSliceRight
                    # x[a:] = v      SetSliceLeft
                    args = []
                    left = right = None
                    is_slice = False
                    if consume(':'):
                        is_slice = True
                        if consume(']'):
                            pass
                        else:
                            right = parse_expression(ctx)
                            args.append(right)
                            expect(']')
                    elif consume(']'):
                        pass
                    else:
                        left = parse_expression(ctx)
                        args.append(left)
                        if consume(':'):
                            is_slice = True
                            if consume(']'):
                                pass
                            else:
                                right = parse_expression(ctx)
                                args.append(right)
                                expect(']')
                        elif consume(']'):
                            pass
                        else:
                            expect(',')
                            while not consume(']'):
                                args.append(parse_expression(ctx))
                                if not consume(','):
                                    expect(']')
                                    break
                    assign = False
                    if consume('='):
                        assign = True
                        args.append(parse_expression(ctx))
                    if is_slice:
                        method_name = (
                            'SliceAll' if left is None and right is None else
                            'SliceLeft' if right is None else
                            'SliceRight' if left is None else
                            'Slice'
                        )
                        if assign:
                            method_name = 'Set' + method_name
                    elif assign:
                        method_name = 'SetItem'
                    else:
                        method_name = 'GetItem'
                    expr = ir.MethodCall(token, expr, method_name, args)
                    continue
                break
            return expr

        def parse_primary(ctx):
            token = peek()

            if consume('def'):
                next_int = intgen[0]
                intgen[0] += 1
                lambda_name = f'lambda#{next_int}'
                return_type = expect_type()
                capture_params = []
                if consume('['):
                    while not consume(']'):
                        if at('NAME') and at('NAME', 1):
                            type_ = expect_type()
                        else:
                            type_ = 'var'
                        capture_params.append(ir.Parameter(
                            peek(), type_, expect_non_exported_name()))
                        if not consume(','):
                            expect(']')
                            break
                params = parse_params()
                body = parse_block(ctx)
                ctx.defs.append(ir.FunctionDefinition(
                    token,
                    return_type,
                    lambda_name,
                    capture_params + params,
                    body))
                return ir.FunctionCall(token, 'Closure', [
                    ir.ListDisplay(token, [
                        ir.Name(p.token, p.name) for p in capture_params
                    ]),
                    ir.Name(token, lambda_name),
                ])

            if consume('('):
                with skipping_newlines(True):
                    expr = parse_expression(ctx)
                    expect(')')
                return expr

            if consume('['):
                exprs = []
                with skipping_newlines(True):
                    while not consume(']'):
                        exprs.append(parse_expression(ctx))
                        if not consume(','):
                            expect(']')
                            break
                return ir.ListDisplay(token, exprs)

            if consume('{'):
                with skipping_newlines(True):
                    if consume('}'):
                        return ir.FunctionCall(token, 'Set', [])
                    if consume(':'):
                        expect('}')
                        return ir.FunctionCall(token, 'Map', [])
                    key = parse_expression(ctx)
                    if consume(':'):
                        value = parse_expression(ctx)
                        pairs = [ir.ListDisplay(token, [key, value])]
                        if consume(','):
                            while not consume('}'):
                                key = parse_expression(ctx)
                                expect(':')
                                value = parse_expression(ctx)
                                pairs.append(ir.ListDisplay(token, [key, value]))
                                if not consume(','):
                                    expect('}')
                                    break
                        else:
                            expect('}')
                        return ir.FunctionCall(token, 'MapFromPairs', [
                            ir.ListDisplay(token, pairs)
                        ])
                    args = [key]
                    if consume(','):
                        while not consume('}'):
                            args.append(parse_expression(ctx))
                            if not consume(','):
                                expect('}')
                                break
                    else:
                        expect('}')
                    return ir.FunctionCall(token, 'SetFromList', [ir.ListDisplay(token, args)])

            if consume('null'):
                type_ = 'var'
                if consume('('):
                    type_ = expect_type()
                    expect(')')
                return ir.NullLiteral(token, type_)

            if consume('true'):
                return ir.BoolLiteral(token, True)

            if consume('false'):
                return ir.BoolLiteral(token, False)

            if at('INT'):
                value = expect('INT').value
                return ir.IntLiteral(token, value)

            if at('FLOAT'):
                value = expect('FLOAT').value
                return ir.DoubleLiteral(token, value)

            if at('NAME') or (at('try') and at('.', 1) and at('NAME', 2)):
                try_ = consume('try')
                if try_:
                    expect('.')
                name = expect_maybe_exported_name()
                if try_:
                    name = f'{name}%try'
                if consume('++'):
                    return ir.SetName(token, name, ir.MethodCall(
                        token, ir.Name(token, name), 'Add', [ir.IntLiteral(token, 1)]))
                if consume('--'):
                    return ir.SetName(token, name, ir.MethodCall(
                        token, ir.Name(token, name), 'Sub', [ir.IntLiteral(token, 1)]))
                if consume('='):
                    expr = parse_expression(ctx)
                    return ir.SetName(token, name, expr)
                if at('('):
                    args = parse_args(ctx)
                    return ir.FunctionCall(token, name, args)
                return ir.Name(token, name)

            if at('STRING'):
                return ir.StringLiteral(token, expect('STRING').value)

            raise Error([token], 'Expected expression')

        def at_variable_definition():
            return (at('final') or at('auto') or at('NAME')) and at('NAME', 1)

        def parse_variable_definition(ctx):
            token = peek()
            if consume('final'):
                final = True
                vartype = None
            elif consume('auto'):
                final = False
                vartype = None
            else:
                final = False
                vartype = expect_type()
            name = expect_non_exported_name()
            value = parse_expression(ctx) if consume('=') else None
            expect_delim()
            if final and value is None:
                raise Error(
                    [self.token],
                    'final variables definitions must specify an expression')
            return ir.VariableDefinition(token, final, vartype, name, value)

        def parse_args(ctx, opener='(', closer=')'):
            args = []
            with skipping_newlines(True):
                expect(opener)
                while not consume(closer):
                    args.append(parse_expression(ctx))
                    if not consume(','):
                        expect(closer)
                        break
                return args

        def parse_params():
            params = []
            expect('(')
            with skipping_newlines(True):
                while not consume(')'):
                    paramtoken = peek()
                    paramtype = expect_type()
                    paramname = expect_non_exported_name()
                    params.append(ir.Parameter(paramtoken, paramtype, paramname))
                    if not consume(','):
                        expect(')')
                        break
                return params

        return parse_program()


tok = lexer.lex(Source('<dummy>', 'dummy'))[0]

argparser = argparse.ArgumentParser()
argparser.add_argument('kfile')
argparser.add_argument('--out-file', '-o', default=None)
argparser.add_argument('--opt', '-O', type=int, default=1)
argparser.add_argument('--debug', '-g', action='store_true', default=False)
argparser.add_argument('--no-debug', dest='debug', action='store_false')
argparser.add_argument('--test', '-t', action='store_true', default=False)


def find_tests(local_prefix):
    lib = os.path.join(_scriptdir, 'lib')
    root = os.path.join(lib, local_prefix.replace('.', os.sep))
    if root.endswith('_test') and os.path.isfile(root + '.k'):
        yield local_prefix
    for base, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('_test.k'):
                yield os.path.relpath(
                    os.path.join(base, filename[:-2]), lib
                ).replace(os.sep, '.')


def main():
    try:
        args = argparser.parse_args()
        builtins_node = parser.parse(Source('<builtin>', BUILTINS), None, None)

        if args.test:
            tests = sorted(find_tests(args.kfile))
            sb = FractalStringBuilder(0)
            for i, t in enumerate(tests):
                sb += f'import {t} as test{i}'

            sb += 'void main() {'
            sbb = sb.spawn(1)
            for i, t in enumerate(tests):
                sbb += f'STDERR.write("Testing {t}... ")'
                sbb += f'test{i}.main()'
                sbb += f'STDERR.write("pass\\n")'
            sb += 'STDERR.write("All tests pass!\\n")'
            sb += '}'

            source = Source('<test>', str(sb))
        else:
            with open(args.kfile) as f:
                data = f.read()
            source = Source(args.kfile, data)

        node = parser.parse(source, '#', builtins_node.env)
        program = ir.Program(
            node.token,
            node.definitions + builtins_node.definitions,
            node.env)

        c_src = Cee.translate(program)

        env = program.env

        with open(os.path.join(_scriptdir, 'c', 'generated_main.c'), 'w') as f:
            f.write(c_src)

        run_compiler(env, args.out_file, args.opt, args.debug)
    except Error as e:
        sys.stderr.write(f'{e}\n')
        exit(1)


def run_compiler(env, out_file, opt, debug):
    if sys.platform == 'win32':
        # TODO: pass in opt flag to windows compiler
        run_compiler_for_windows(env, out_file)
    elif sys.platform == 'darwin':
        run_compiler_for_osx(env, out_file, opt, debug)
    elif sys.platform == 'linux':
        run_compiler_for_linux(env, out_file, opt, debug)
    else:
        raise TypeError('Unsupported platform %s' % (sys.platform, ))


def run_compiler_for_windows(env, out_file):
    # TODO: Support different versions of visual studio, and handle
    # missing C compiler more gracefully.
    vcvars_path = r'C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat'
    exe_name = os.path.join(_scriptdir, 'a.exe')
    base = os.path.join(_scriptdir, 'c')
    srcs = env['@vars']['C_SOURCES']
    srcsstr = ' '.join(f'"{os.path.join(base, p)}"' for p in srcs)

    if out_file is None:
        out_file = 'a.exe'

    if not out_file.endswith('.exe'):
        out_file += '.exe'

    compile_cmd = (
        f'"{vcvars_path}" x64 2> NUL > NUL && '
        f'cl {srcsstr} /Fe{exe_name} /WX'
    )
    quiet_compile = f'{compile_cmd} 2> NUL > NUL'

    try:
        subprocess.run(
            f'{quiet_compile} && move {exe_name} {out_file} > NUL',
            check=True,
            shell=True,
        )
    except subprocess.CalledProcessError:
        exit(subprocess.run(compile_cmd, shell=True).returncode)


def run_compiler_for_osx(env, out_file, opt, debug):

    framework_flags = ''
    for framework in env['@vars']['APPLE_FRAMEWORKS']:
        framework_flags += f' -framework {framework}'

    warning_flags = '-Wno-long-long'

    run_compiler_for_unix(
        'clang',
        env,
        out_file,
        opt,
        debug,
        flags=f'{framework_flags} {warning_flags} -flto -fobjc-arc')


def run_compiler_for_linux(env, out_file, opt, debug):
    warning_flags = (
        '-Wno-unused-but-set-variable '
    )
    run_compiler_for_unix('gcc', env, out_file, opt, debug, warning_flags)


def run_compiler_for_unix(cc, env, out_file, opt, debug, flags=''):
    if out_file is None:
        out_file = 'a.out'

    base = os.path.join(_scriptdir, 'c')
    srcs = env['@vars']['C_SOURCES']
    srcsstr = ' '.join(os.path.join(base, p) for p in srcs)

    dbg = '-g ' if debug else ''
    compile_cmd = (
        f'{cc} -Wall -Werror -Wpedantic -std=c89 -O{opt} {dbg}'
        f'-Wno-unused-function -Wno-unused-variable '
        f'{srcsstr} -o {out_file} {flags}'
    )

    try:
        subprocess.run(compile_cmd, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        exit(e.returncode)


if __name__ == '__main__':
    main()
