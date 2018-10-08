from typing import NamedTuple, Tuple, List, Union, Optional, Callable, Iterable
import abc
import argparse
import contextlib
import os
import re
import sys
import typing


class Namespace(object):
    def __init__(self, callback):
        self.__name__ = callback.__name__
        self._fields = set()
        callback(self)

    def __call__(self, item, name=None):
        name = item.__name__ if name is None else name
        self._fields.add(name)
        setattr(self, name, item)
        return item

    def __iadd__(self, other):
        for name in other._fields:
            setattr(self, name, getattr(other, name))
        return self


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

    @ns
    class Multimethod:
        def __init__(self, name):
            self.__name__ = name
            self.implementations = []

        def __repr__(self):
            return f'<multimethod {self.name}>'

        def __call__(self, *args):
            for types, f in self.implementations:
                if (len(types) <= len(args) and
                        all(isinstance(arg, t) for arg, t in zip(args, types))):
                    return f(*args)
            raise TypeError('Multimethod no implementation found for '
                            f'{[type(arg) for arg in args]}')

        def define(self, *types):
            def wrapper(f):
                self.implementations.append((types, f))
                return self
            return wrapper


@Namespace
def lexutil(ns):
    @ns
    class Source(NamedTuple):
        filename: str
        data: str

    @ns
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
            return f'on line {lineno}\n{line}\n{spaces}*\n'

    @ns
    class Error(Exception):
        def __init__(self, tokens: Iterable[Token], message: str) -> None:
            super().__init__(''.join(token.info for token in tokens) + message)

    SYMBOLS = [
        '\n',
        '||', '&&',
        ';', '#', '?', ':',
        '.', ',', '!', '@', '^', '&', '+', '-', '/', '%', '*', '.', '=', '==', '<',
        '>', '<=', '>=', '(', ')', '{', '}', '[', ']',
    ]

    KEYWORDS = {
        'is', 'not', 'null', 'true', 'false', 'new',
        'extern',
        'class', 'trait',
        'if', 'else', 'while', 'break', 'continue', 'return',
    }

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

    whitespace_and_comment_pattern = RegexPattern(r'(?:[ \t\r]|//.*\n)+')

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
        string_pattern_regex,
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

    _lexer = make_simple_lexer(keywords=KEYWORDS, symbols=SYMBOLS)

    @ns
    def lex(source):
        return _lexer.lex(source)


@Namespace
def util(ns):
    ns += typeutil
    ns += lexutil

    @ns
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


@Namespace
def ast0(ns):
    """ast0 is the lowest level syntax tree that maps directly
    to the needed subset C code as close as possible.
    """
    List = util.List
    Optional = util.Optional

    @ns
    class Node(util.Node):
        pass

    @ns
    class Definition(Node):
        pass

    @ns
    class Statement(Node):
        pass

    @ns
    class Expression(Node):
        pass

    @ns
    class Block(Statement):
        fields = (
            ('statements', List[Statement]),
        )

    @ns
    class TranslationUnit(Node):
        fields = (
            ('definitions', Definition),
        )

    @ns
    class Blob(Definition):
        fields = (
            ('text', str),
        )

    @ns
    class Parameter(Node):
        fields = (
            ('type', str),
            ('name', str),
        )

    @ns
    class FunctionDefinition(Definition):
        fields = (
            ('return_type', str),
            ('name', str),
            ('params', List[Parameter]),
            ('body', Optional[Block]),
        )

    @ns
    class Field(Node):
        fields = (
            ('type', str),
            ('name', str),
        )

    @ns
    class StructDefinition(Definition):
        fields = (
            ('name', str),
            ('fields', List[Field]),
        )

    def translate(translation_unit):
        src = util.FractalStringBuilder()
        blobs = src.spawn()
        typedefs = src.spawn()
        structs = src.spawn()
        funcprotos = src.spawn()
        vardefs = src.spawn()
        funcdefs = src.spawn()

        translate_definition = util.Multimethod('translate_definition')

        @translate_definition.define(Blob)
        def translate_definition(blob):
            blobs += blob.text

        @translate_definition.define(StructDefinition)
        def translate_definition(struct):
            typedefs += f'typedef struct {struct.name} {struct.name};'
            structs += f'struct {struct.name} ' '{'
            for field in struct.fields:
                structs += f'  {field.type} {field.name};'
            structs += '}'

        @translate_definition.define(FunctionDefinition)
        def translate_definition(fn):
            args = ', '.join(f'{p.type} {p.name}' for p in fn.params)
            proto = f'{fn.return_type} {fn.name}({args})'
            funcprotos += f'{proto};'
            if fn.body:
                raise TypeError('TODO')

        return str(src)


