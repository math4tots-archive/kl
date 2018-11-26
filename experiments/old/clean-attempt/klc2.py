from typing import NamedTuple, Tuple, List, Union, Optional, Callable, Iterable
import abc
import argparse
import contextlib
import os
import re
import shutil
import string
import subprocess
import sys
import typing

_scriptdir = os.path.dirname(os.path.realpath(__file__))


SYMBOLS = [
    '\n',
    '||', '&&',
    ';', '#', '?', ':', '!', '++', '--',
    '.', ',', '!', '@', '^', '&', '+', '-', '/', '%', '*', '.', '=', '==', '<',
    '>', '<=', '>=', '!=', '(', ')', '{', '}', '[', ']',
]

KEYWORDS = {
    'is', 'not', 'null', 'true', 'false', 'new', 'and', 'or', 'in',
    'inline', 'extern', 'class', 'trait', 'final', 'def', 'auto',
    'for', 'if', 'else', 'while', 'break', 'continue', 'return',
    'with', 'from', 'import', 'as',
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


class Object:
    "Dummy object class to attach misc attributes to"

    @contextlib.contextmanager
    def _bind(self, name, value):
        """temporarily rebind an attribute of this object to a different
        value
        """
        oldvalue = getattr(self, name)
        setattr(self, name, value)
        yield
        setattr(self, name, oldvalue)

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


class Source(NamedTuple):
    filename: str
    data: str


class Token:
    __slots__ = ['type', 'value', 'source', 'i']

    def __init__(self,
                 type: str,
                 value: object = None,
                 source: Optional[Source] = None,
                 i: Optional[int] = None) -> None:
        self.type = type
        self.value = value
        self.source = source
        self.i = i

    def _key(self) -> Tuple[str, object]:
        return self.type, self.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Token) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return f'Token({repr(self.type)}, {repr(self.value)})'

    @property
    def lineno(self) -> int:
        assert self.source is not None
        assert self.i is not None
        return self.source.data.count('\n', 0, self.i) + 1

    @property
    def colno(self) -> int:
        assert self.source is not None
        assert self.i is not None
        return self.i - self.source.data.rfind('\n', 0, self.i)

    @property
    def line(self) -> str:
        assert self.source is not None
        assert self.i is not None
        s = self.source.data
        a = s.rfind('\n', 0, self.i) + 1
        b = s.find('\n', self.i)
        if b == -1:
            b = len(s)
        return s[a:b]

    @property
    def info(self) -> str:
        line = self.line
        colno = self.colno
        lineno = self.lineno
        spaces = ' ' * (colno - 1)
        return f'{self.source.filename} on line {lineno}\n{line}\n{spaces}*\n'


class Error(Exception):
    def __init__(self, tokens: Iterable[Token], message: str) -> None:
        super().__init__(''.join(token.info for token in tokens) + message)


def _make_lexer():

    class Pattern(abc.ABC):
        @abc.abstractmethod
        def match(self, source: Source, i: int) -> Optional[Tuple[Token, int]]:
            pass

    class RegexPattern(Pattern):
        def __init__(
                self,
                regex: Union[typing.Pattern[str], str],
                *,
                type: Optional[str] = None,
                type_callback: Callable[[str], str] = lambda value: value,
                value_callback: Callable[[str], object] = lambda x: x) -> None:
            if isinstance(regex, str):
                regex = re.compile(regex)

            if type is not None:
                type_ = str(type)  # for mypy
                type_callback = lambda _: type_

            self.regex = regex
            self.type_callback = type_callback
            self.value_callback = value_callback

        def match(self, source: Source, i: int) -> Optional[Tuple[Token, int]]:
            m = self.regex.match(source.data, i)
            if m is None:
                return None

            raw_value = m.group()
            type_ = self.type_callback(raw_value)
            value = self.value_callback(raw_value)
            return Token(type_, value, source, i), m.end()

    whitespace_and_comment_pattern = RegexPattern(r'(?:[ \t\r]|//.*(?=\n))+')

    class Lexer:
        def __init__(
                self,
                patterns: List[Pattern],
                *,
                ignore_pattern: Pattern,
                filter: Callable[[Token], Optional[Token]] = lambda token: token
        ) -> None:
            self.patterns = patterns
            self.ignore_pattern = ignore_pattern
            self.filter = filter

        def lex(self, source: Union[Source, str]) -> List[Token]:
            if isinstance(source, str):
                source = Source('<string>', source)

            ignore_pattern = self.ignore_pattern
            i = 0
            s = source.data
            tokens = []

            while True:
                while True:
                    match = ignore_pattern.match(source, i)
                    if match is None:
                        break
                    _, i = match

                if i >= len(s):
                    break

                for pattern in self.patterns:
                    match = pattern.match(source, i)
                    if match is not None:
                        unfiltered_token, i = match
                        token = self.filter(unfiltered_token)
                        if token is not None:
                            tokens.append(token)
                        break
                else:
                    token = Token('ERR', None, source, i)
                    raise Error([token], 'Unrecognized token')

            tokens.append(Token('EOF', None, source, i))
            return tokens

    class MatchingBracesSkipSpacesFilter:
        """A lexer filter for ignoring newlines that appear inside parentheses
        or square brackets.
        """

        def __init__(self) -> None:
            self.stack: List[Token] = []

        def should_skip_newlines(self) -> bool:
            return bool(self.stack and self.stack[-1].type != '{')

        def __call__(self, token: Token) -> Optional[Token]:
            if token.type in ('{', '[', '('):
                self.stack.append(token)

            if token.type in ('}', ']', ')'):
                self.stack.pop()

            if token.type == '\n' and self.should_skip_newlines():
                return None

            return token

    def make_symbols_pattern(symbols: Iterable[str]) -> Pattern:
        return RegexPattern('|'.join(map(re.escape, reversed(sorted(symbols)))))

    def make_keywords_pattern(keywords: Iterable[str]) -> Pattern:
        return RegexPattern('|'.join(r'\b' + kw + r'\b' for kw in keywords))

    name_pattern = RegexPattern(
        '\w+', type='NAME', value_callback=lambda value: value)

    def string_pattern_value_callback(value: str) -> str:
        return str(eval(value))  # type: ignore

    string_pattern_regex = '|'.join([
        r'(?:r)?"""(?:(?:\\.)|(?!""").)*"""',
        r"(?:r)?'''(?:(?:\\.)|(?!''').)*'''",
        r'(?:r)?"(?:(?:\\.)|(?!").)*"',
        r"(?:r)?'(?:(?:\\.)|(?!').)*'",
    ])

    string_pattern = RegexPattern(
        re.compile(string_pattern_regex, re.DOTALL),
        type='STRING',
        value_callback=string_pattern_value_callback)

    float_pattern = RegexPattern(
        r'\d+\.\d*|\.\d+', type='FLOAT', value_callback=eval)  # type: ignore
    int_pattern = RegexPattern(
        r'\d+', type='INT', value_callback=eval)  # type: ignore

    def make_simple_lexer(*, keywords: Iterable[str], symbols: Iterable[str]):
        keywords_pattern = make_keywords_pattern(keywords)
        symbols_pattern = make_symbols_pattern(symbols)
        return Lexer(
            [
                string_pattern,
                keywords_pattern,
                float_pattern,
                int_pattern,
                name_pattern,
                symbols_pattern,
            ],
            ignore_pattern=whitespace_and_comment_pattern)

    return make_simple_lexer(keywords=KEYWORDS, symbols=SYMBOLS)


_lexer = _make_lexer()


def lex(source):
    return _lexer.lex(source)


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


class SetType(FakeType):
    def __init__(self, subtype):
        self.subtype = subtype

    def __getitem__(self, subtype):
        return SetType(subtype)

    def __instancecheck__(self, obj):
        return (
            isinstance(obj, set) and
            all(isinstance(x, self.subtype) for x in obj))

    def __repr__(self):
        return 'Set[%s]' % (self.subtype.__name__, )


Set = SetType(object)


class MapType(FakeType):
    def __init__(self, ktype, vtype):
        self.ktype = ktype
        self.vtype = vtype

    def __getitem__(self, kvtypes):
        ktype, vtype = kvtypes
        return MapType(ktype, vtype)

    def __instancecheck__(self, obj):
        return (
            isinstance(obj, set) and
            all(isinstance(k, self.ktype) and
                isinstance(v, self.vtype) for k, v in obj.items()))

    def __repr__(self):
        return 'Map[%s,%s]' % (self.ktype.__name__, self.vtype.__name__)


Map = MapType(object, object)


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


class BaseVariableDefinition(Node):
    fields = (
        ('type', str),
        ('name', str),
    )


class CParameter(BaseVariableDefinition):
    pass


class CField(BaseVariableDefinition):
    pass


class Parameter(BaseVariableDefinition):
    pass


class Field(BaseVariableDefinition):
    pass


@Namespace
def C(ns):
    """
    C AST

    This is an intermediate representation used to more
    conveniently generate C source

    C types are stored as simple strings without much structure.
    Function pointer types are unfortunately not supported
    due to the fact that they can't be declared very simply.

    Supported types are:
        primitive types (the name is stored as str)
        struct types (the name is stored as str)
        pointer types (name of base type with trailing asterisks)
    """

    @ns
    class N(Node):
        pass

    @ns
    class Global(N):
        pass

    @ns
    class Statement(N):
        pass

    @ns
    class Expression(N):
        def __call__(self, ctx):
            with ctx._bind('src', None):
                expr = self.translate(ctx)
                assert isinstance(expr, str), expr
                return expr

    @ns
    class Block(Statement):
        fields = (
            ('statements', List[Statement]),
        )

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            seen_non_vardef = False
            for statement in self.statements:
                if (isinstance(statement, LocalVariableDefinition) and
                        seen_non_vardef):
                    raise Error(
                        [statement.token],
                        'All variable definitions must come at the '
                        'beginning of a block')
                if not isinstance(statement, LocalVariableDefinition):
                    seen_non_vardef = True

        def translate(self, ctx):
            ctx.src += '{'
            src = ctx.src.spawn(1)
            ctx.src += '}'

            with ctx._bind('src', src):
                for statement in self.statements:
                    statement.translate(ctx)

    @ns
    class LocalVariableDefinition(BaseVariableDefinition, Statement):
        def translate(self, ctx):
            # I want to initialize all local variables
            # to minimize potential chance of unexpected behavior
            if self.type in ('size_t', 'int', 'long', 'char'):
                zero = '0'
            elif self.type.endswith('*'):
                zero = 'NULL'
            else:
                raise Error([self.token], f'Unrecognized type {self.type}')
            ctx.src += f'{self.type} {self.name} = {zero};'

    @ns
    class Return(Statement):
        fields = (
            ('expression', Expression),
        )

        def translate(self, ctx):
            ctx.src += f'return {self.expression(ctx)};'

    @ns
    class ExpressionStatement(Statement):
        fields = (
            ('expression', Expression),
        )

        def translate(self, ctx):
            ctx.src += f'{self.expression(ctx)};'

    @ns
    class Prefix(Expression):
        fields = (
            ('operator', str),
            ('expression', Expression),
        )

        def translate(self, ctx):
            return f'({self.operator} {self.expression(ctx)})'

    @ns
    class Postfix(Expression):
        fields = (
            ('expression', Expression),
            ('operator', str),
        )

        def translate(self, ctx):
            return f'({self.expression(ctx)} {self.operator})'

    @ns
    class Binop(Expression):
        fields = (
            ('left', Expression),
            ('operator', str),
            ('right', Expression),
        )

        def translate(self, ctx):
            return f'({self.left(ctx)} {self.operator} {self.right(ctx)})'

    @ns
    class StringLiteral(Expression):
        fields = (
            ('value', str),
        )

        def translate(self, ctx):
            # TODO: Come up with better way to do this
            value = (
                self.value
                    .replace('\\', '\\\\')
                    .replace('"', '\\"')
                    .replace('\n', '\\n')
                    .replace('\r', '\\r')
                    .replace('\t', '\\t')
            )
            return f'"{value}"'

    @ns
    class IntLiteral(Expression):
        fields = (
            ('value', int),
        )

        def translate(self, ctx):
            return f'{self.value}L'

    @ns
    class FloatLiteral(Expression):
        fields = (
            ('value', float),
        )

        def translate(self, ctx):
            return str(self.value)

    @ns
    class Name(Expression):
        fields = (
            ('name', str),
        )

        def translate(self, ctx):
            return self.name

    @ns
    class SetName(Expression):
        fields = (
            ('name', str),
            ('expression', Expression),
        )

        def translate(self, ctx):
            return f'({self.name} = {self.expression(ctx)})'

    @ns
    class FunctionCall(Expression):
        fields = (
            ('name', str),
            ('args', List[Expression]),
        )

        def translate(self, ctx):
            args = ','.join(e(ctx) for e in self.args)
            return f'({self.name}({args}))'

    @ns
    class GetItem(Expression):
        fields = (
            ('expression', Expression),
            ('index', Expression),
        )

        def translate(self, ctx):
            expr = self.expression(ctx)
            index = self.index(ctx)
            return f'({expr}[{index}])'

    @ns
    class SetItem(Expression):
        fields = (
            ('expression', Expression),
            ('index', Expression),
            ('value', Expression),
        )

        def translate(self, ctx):
            expr = self.expression(ctx)
            index = self.index(ctx)
            value = self.value(ctx)
            return f'({expr}[{index}] = {value})'

    @ns
    class DotGet(Expression):
        fields = (
            ('expression', Expression),
            ('name', str),
        )

        def translate(self, ctx):
            expr = self.expression(ctx)
            return f'({expr}.{self.name})'

    @ns
    class DotSet(Expression):
        fields = (
            ('expression', Expression),
            ('name', str),
            ('value', Expression),
        )

        def translate(self, ctx):
            expr = self.expression(ctx)
            value = self.value(ctx)
            return f'({expr}.{self.name} = {value})'

    @ns
    class ArrowGet(Expression):
        fields = (
            ('expression', Expression),
            ('name', str),
        )

        def translate(self, ctx):
            expr = self.expression(ctx)
            return f'({expr}->{self.name})'

    @ns
    class ArrowSet(Expression):
        fields = (
            ('expression', Expression),
            ('name', str),
            ('value', Expression),
        )

        def translate(self, ctx):
            expr = self.expression(ctx)
            value = self.value(ctx)
            return f'({expr}->{self.name} = {value})'

    @ns
    class TranslationUnit(N):
        fields = (
            ('name', str),
            ('globals', List[Global]),
        )

        def write_out(self, basedir=None):
            if basedir is None:
                basedir = os.path.join(_scriptdir, 'gen')
            include_dir = os.path.join(basedir, 'include')
            source_dir = os.path.join(basedir, 'source')
            include_fn = os.path.join(include_dir, f'{self.name}.h')
            source_fn = os.path.join(source_dir, f'{self.name}.c')
            hdr = FractalStringBuilder(0)
            src = FractalStringBuilder(0)
            self.translate(hdr, src)
            hdr = str(hdr)
            src = str(src)
            with open(include_fn, 'w') as f:
                f.write(hdr)
            with open(source_fn, 'w') as f:
                f.write(src)

        def translate(self, hdr, src):
            ctx = Object()
            hdr += f'#ifndef {self.name.upper()}_H'
            hdr += f'#define {self.name.upper()}_H'
            ctx.includes = hdr.spawn()
            ctx.typedefs = hdr.spawn()
            ctx.protos = hdr.spawn()
            ctx.structs = hdr.spawn()
            ctx.externvars = hdr.spawn()
            hdr += f'#endif/*{self.name.upper()}_H*/'

            src += f'#include "{self.name}.h"'
            ctx.vars = src.spawn()
            ctx.src = src.spawn()

            for g in self.globals:
                g.translate(ctx)

    @ns
    class Include(Global):
        fields = (
            ('name', str),
        )

        def translate(self, ctx):
            ctx.includes += f'#include "{self.name}"'

    @ns
    class FunctionDefinition(Global):
        fields = (
            ('return_type', str),
            ('name', str),
            ('parameters', List[CParameter]),
            ('body', Optional[Block]),
        )

        @property
        def proto(self):
            params = ', '.join(f'{p.type} {p.name}' for p in self.parameters)
            return f'{self.return_type} {self.name}({params})'

        def translate(self, ctx):
            proto = self.proto
            ctx.protos += f'{proto};'

            if self.body is not None:
                ctx.src += proto
                fctx = Object()
                fctx.src = ctx.src.spawn()
                self.body.translate(fctx)

    @ns
    class StructDefinition(Global):
        fields = (
            ('name', str),
            ('fields', List[CField]),
        )

        def translate(self, ctx):
            ctx.typedefs += f'typedef struct {name} {name};'
            ctx.structs += f'struct {name} ' '{'
            out = ctx.structs.spawn(1)
            for field in self.fields:
                out += f'{field.type} {field.name};'
            ctx.structs += '};'

    @ns
    class GlobalVariableDefinition(Global, BaseVariableDefinition):
        def translate(self, ctx):
            proto = f'{self.type} {self.name}'
            ctx.externvars += f'extern {proto};'
            ctx.vars += f'{proto};'


@Namespace
def D(ns):
    """Datastructures that describe information about global symbols.
    """

    @ns
    class Context:
        def __init__(self):
            self.d = dict()

        def add(self, node):
            assert isinstance(node, N), node
            if node.name in self.d:
                raise Error(
                    [node.token, self.d[node.name].token],
                    f'Duplicate definition of {node.name}')
            self.d[node.name] = node

    class N(Node):
        pass

    @ns
    class Function(N):
        fields = (
            ('extern', bool),
            ('return_type', str),
            ('name', str),
            ('parameters', List[Parameter]),
        )

    @ns
    class Trait(N):
        fields = (
            ('name', str),
            ('traits', List[str]),
            ('methods', List[str]),
            ('static_methods', List[str]),
        )

    @ns
    class Class(N):
        fields = (
            ('extern', bool),
            ('name', str),
            ('traits', List[str]),
            ('fields', List[Field]),
            ('methods', List[str]),
            ('static_methods', List[str]),
        )

    @ns
    class GlobalVariable(N):
        fields = (
            ('extern', bool),
            ('type', str),
            ('name', str),
        )


@Namespace
def AK(ns):
    """Annotated K AST
    These can generate a C AST

    All names should be fully qualified in this AST.
    All expressions should be annotated with their type

    Note, that some names (e.g. class names, function names,
    global variable names) will contain '.' characters if they
    belong to a package. These names are encoded when translated to C.
    """

    @ns
    class N(Node):
        pass

    @ns
    class Global(N):
        pass

    @ns
    class Statement(N):
        pass

    @ns
    class Expression(N):
        pass

    @ns
    class BaseVariableDefinition(N):
        fields = (
            ('type', str),
            ('name', str),
        )

    @ns
    class LocalVariableDefinition(BaseVariableDefinition):
        pass

    @ns
    class Block(Statement):
        fields = (
            ('statements', Statement),
        )

    @ns
    class TranslationUnit(N):
        fields = (
            ('name', str),
            ('globals', List[Global]),
        )

    @ns
    class FunctionDefinition(Global):
        fields = (
            ('return_type', str),
            ('name', str),
            ('parameters', List[Parameter]),
            ('body', Optional[Block]),
        )

    @ns
    class ClassDefinition(Global):
        fields = (
            ('name', str),
            ('methods', List[str]),
        )

@Namespace
def RK(ns):
    """Raw K AST
    These can generate Annotated K AST (AK) when a populated
    global D.Context is provided.

    All names should still be fully qualified,
    but most expressions will not be annotated with their types.
    """
    @ns
    class N(Node):
        pass

    @ns
    class Global(N):
        pass

    @ns
    class Statement(N):
        pass

    @ns
    class Expression(N):
        pass

    @ns
    class Block(Statement):
        fields = (
            ('statements', List[Statement]),
        )

    @ns
    class TranslationUnit(N):
        fields = (
            ('name', str),
            ('globals', List[Global]),
        )

    @ns
    class Import(Global):
        fields = (
            ('name', str),  # name of module to import
        )

    @ns
    class ClassDefinition(Global):
        fields = (
            # All other information should be in D.Context
            ('name', str),
        )

    @ns
    class TraitDefinition(Global):
        fields = (
            # All other information should be in D.Context
            ('name', str),
        )

    @ns
    class GlobalVariableDefinition(Global):
        fields = (
            # All other information should be in D.Context
            ('name', str),
            ('expression', Optional[Expression]),
        )

    @ns
    class FunctionDefinition(Global):
        fields = (
            # All other information should be in D.Context
            ('name', str),
            ('body', Optional[Block]),
        )

    @ns
    class ExpressionStatement(Statement):
        fields = (
            ('expression', Expression),
        )

    @ns
    class Return(Statement):
        fields = (
            ('expression', Expression),
        )

    @ns
    class IntLiteral(Expression):
        fields = (
            ('value', int),
        )

    @ns
    class StringLiteral(Expression):
        fields = (
            ('value', str),
        )

    @ns
    class FunctionCall(Expression):
        fields = (
            ('name', str),
            ('args', List[Expression]),
        )

_safe_chars = set(string.ascii_lowercase + string.ascii_uppercase) - {'Z'}


def encode(name):
    """Encode a name so that it's safe to use as a C symbol, and
    prefixed with 'KLCN' to reduce chance of conflict with existing name

    Characters allowed in name:
        digits 0-9
        lower case letters a-z
        upper case letters A-Z
        underscore _
        special characters:
            dot          (.)
            dollar sign  ($)
            percent      (%)
            hash         (#)


    The special characters are to be used by the compiler for special
    generated names (e.g. auto-generated functions).

    The encoding mostly allows letters and digits to be themselves,
    except capital 'Z' is used as an escape character, to encode all
    other kinds of characters.
    """
    chars = ['KLCN']
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
        elif c in _safe_chars:
            chars.append(c)
        else:
            raise Error([], f'Invalid character {c} in name {name}')
    return ''.join(chars)


def parse(source, ctx: D.Context, tu_name='#'):
    env = {
        '@cache': dict(),
        '@queued': set(),
        '@ctx': ctx,
        '@todo': [(source, tu_name)],
    }

    translation_units = []

    while env['@todo']:
        source, name = env['@todo'].pop()
        translation_units.append(parse_one_source(source, name, env))

    return translation_units


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
            symbol.add(expect('NAME').value)
        elif at('(') or at('[') or at('{'):
            skip_to_matching()
        else:
            gettok()

    return symbols


def parse_one_source(source, tu_name, env):
    ""
    """
    lname = long name
        either a MODULE qualified name, or
        sname
    sname = short name
        NAME, that's also potentially qualified if
        it's one of local_symbols
    """
    tokens = lex(source)
    i = 0
    indent_stack = []
    ctx = env['@ctx']  # D.Context

    # Figure out what symbols are explicitly declared by the user
    # in this file.
    # The following cases will use qualified versions
    # of the names when a local symbol is encountered:
    #   * class or trait name,
    #   * variable name,
    #   * function name
    local_symbols = peek_local_symbols(tokens)

    def qualify(n):
        if tu_name:
            return f'{tu_name}.{n}'
        else:
            return n


    def peek(j=0):
        nonlocal i
        if should_skip_newlines():
            while i < len(tokens) and tokens[i].type == '\n':
                i += 1
        return tokens[min(i + j, len(tokens) - 1)]

    def at(type, j=0):
        tok = peek(j)
        t = tok.type
        if t == 'NAME' and tok.value in import_names:
            return type == 'MODULE'
        else:
            return t == type

    def at_seq(types, j=0):
        return all(at(t, i) for i, t in enumerate(types, j))

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

    def at_lname():
        return at_seq(['MODULE', '.', 'NAME']) or at('NAME')

    def expect_lname():
        if at_lname():
            if at('MODULE'):
                module = import_names[expect('MODULE').value]
                expect('.')
                return module + '.' + expect('NAME').value
            else:
                return expect_sname()
        raise Error([peek()], 'Expected name')

    def at_sname():
        return at('NAME')

    def expect_sname():
        if at_sname():
            name = expect('NAME').value
            if name in local_symbols:
                return qualify(name)
            else:
                return name
        raise Error([peek()], 'Expected short name')

    def at_type():
        return at_lname()

    def expect_type():
        return expect_lname()

    def at_variable_definition():
        return (
            at_seq(['MODULE', '.', 'NAME', 'NAME', '=']) or
            at_seq(['MODULE', '.', 'NAME', 'NAME', '\n']) or
            at_seq(['NAME', 'NAME', '=']) or
            at_seq(['NAME', 'NAME', '\n'])
        )

    def at_function_definition():
        return (
            at_seq(['MODULE', '.', 'NAME', 'NAME', '(']) or
            at_seq(['NAME', 'NAME', '('])
        )

    def at_class_definition():
        return at_seq(['extern', 'class']) or at('class')

    def parse_import():
        token = expect('import')
        parts = [expect('NAME').value]
        while consume('.'):
            parts.append(expect('NAME').value)
        name = '.'.join(parts)
        alias = expect('NAME').value if consume('as') else parts[-1]
        import_names[alias] = name
        expect_delim()
        # TODO: Check if this module needs to be added to the '@todo' stack
        # and do so if it does.
        return RK.Import(token, name)

    def parse_parameters():
        expect('(')
        params = []
        while not consume(')'):
            tok = peek()
            ptype = expect_type()
            pname = expect('NAME').value
            params.append(Parameter(tok, ptype, pname))
            if not consume(','):
                expect(')')
                break
        return params

    def parse_function_definition():
        token = peek()
        return_type = expect_type()
        name = qualify(expect('NAME').value)
        params = parse_parameters()
        body = None if at_delim() else parse_block()
        extern = body is None
        expect_delim()
        ctx.add(D.Function(
            token,
            extern,
            return_type,
            name,
            params,
        ))
        return RK.FunctionDefinition(token, name, body)

    def parse_block():
        token = expect('{')
        consume_delim()
        stmts = []
        while not consume('}'):
            stmts.append(parse_statement())
            consume_delim()
        return RK.Block(token, stmts)

    def parse_variable_definition():
        token = peek()
        vtype = expect_type()
        name = expect_sname()
        expr = parse_expression() if consume('=') else None
        expect_delim()
        return RK.LocalVariableDefinition(token, vtype, name, expr)

    def parse_statement():
        token = peek()
        if at('{'):
            block = parse_block()
            expect_delim()
            return block
        if at_variable_definition():
            return parse_variable_definition()
        expr = parse_expression()
        return RK.ExpressionStatement(token, expr)

    def parse_expression():
        return parse_primary()

    def parse_args():
        expect('(')
        args = []
        while not consume(')'):
            args.append(parse_expression())
            if not consume(','):
                expect(')')
                break
        return args

    def parse_primary():
        token = peek()
        if consume('('):
            expr = parse_expression()
            expect(')')
            return expr
        if at('INT'):
            return RK.IntLiteral(token, expect('INT').value)
        if at('STRING'):
            return RK.StringLiteral(token, expect('STRING').value)
        if at_lname():
            name = expect_lname()
            if at('('):
                args = parse_args()
                return RK.FunctionCall(token, name, args)
            elif consume('='):
                val = parse_expression()
                return RK.SetName(token, name, val)
            else:
                return RK.Name(token, name)
        raise Error([token], 'Expected expression')


    token = peek()
    import_names = dict()
    globals = []
    while at('import'):
        globals.append(parse_import())

    consume_delim()
    while not at('EOF'):
        if at_function_definition():
            globals.append(parse_function_definition())
        else:
            raise Error([peek()], 'Expected global definition')
        consume_delim()

    return RK.TranslationUnit(token, tu_name, globals)


def reset_gen_directory(basedir=None):
    if basedir is None:
        basedir = os.path.join(_scriptdir, 'gen')
    include_dir = os.path.join(basedir, 'include')
    source_dir = os.path.join(basedir, 'source')
    cdir = os.path.join(_scriptdir, 'c')
    idir = os.path.join(_scriptdir, 'include')
    shutil.rmtree(basedir)
    os.makedirs(include_dir, exist_ok=True)
    os.makedirs(source_dir, exist_ok=True)
    for fn in os.listdir(idir):
        shutil.copy(os.path.join(idir, fn), include_dir)
    for fn in os.listdir(cdir):
        shutil.copy(os.path.join(cdir, fn), source_dir)


def build_all(exe_name=None):
    """Only works on linux for now"""
    exe_name = 'a.out' if exe_name is None else exe_name
    include_dir = os.path.join(_scriptdir, 'gen', 'include')
    source_dir = os.path.join(_scriptdir, 'gen', 'source')
    subprocess.run(
        f"gcc -std=c89 -Wall -Werror -Wpedantic -I{include_dir} "
        f"{source_dir}/*.c -o {exe_name}",
        shell=True,
        check=True,
    )

t = lex('')[0]
tu = C.TranslationUnit(t, 'main', [
    C.Include(t, 'klc_runtime.h'),
    C.FunctionDefinition(t, 'int', 'main', [], C.Block(t, [
        C.LocalVariableDefinition(t, 'int', 'x'),
        C.ExpressionStatement(t,
            C.FunctionCall(t, 'printf', [
                C.StringLiteral(t, 'Hello world!\n'),
            ])),
        C.ExpressionStatement(t, C.SetName(t, 'x', C.IntLiteral(t, 18))),
        C.ExpressionStatement(t,
            C.FunctionCall(t, 'printf', [
                C.StringLiteral(t, 'x = %d\n'),
                C.Name(t, 'x'),
            ])),
        C.Return(t, C.IntLiteral(t, 0)),
    ]))
])

src = """
void main() {
    print('Hello world!')
    foo()
}

void foo() {
    print('Inside foo')
}
"""

ctx = D.Context()
print(parse(Source('<test>', src), ctx))

# reset_gen_directory()
# tu.write_out()
# build_all()
print(encode('java.lang.Object'))
