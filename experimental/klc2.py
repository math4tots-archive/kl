from typing import NamedTuple, Tuple, List, Union, Optional, Callable, Iterable
import abc
import argparse
import contextlib
import os
import re
import sys
import typing
import subprocess
import shutil

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
        return object.__getattribute__(self, 'attrs')[key]

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


@Namespace
def C(ns):
    """
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
    class BaseVariableDefinition(N):
        fields = (
            ('type', str),
            ('name', str),
        )

    @ns
    class Parameter(BaseVariableDefinition):
        pass

    @ns
    class Field(BaseVariableDefinition):
        pass

    @ns
    class LocalVariableDefinition(BaseVariableDefinition, Statement):
        def translate(self, ctx):
            ctx.src += f'{self.type} {self.name};'

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
            ('parameters', List[Parameter]),
            ('body', Block),
        )

        @property
        def proto(self):
            params = ', '.join(f'{p.type} {p.name}' for p in self.parameters)
            return f'{self.return_type} {self.name}({params})'

        def translate(self, ctx):
            proto = self.proto
            ctx.protos += f'{proto};'
            ctx.src += proto
            fctx = Object()
            fctx.src = ctx.src.spawn()
            self.body.translate(fctx)

    @ns
    class StructDefinition(Global):
        fields = (
            ('name', str),
            ('fields', List[Field]),
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

reset_gen_directory()
tu.write_out()
build_all()
