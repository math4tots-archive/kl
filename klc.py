"""
From powershell:
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
python .\klc.py > main.c

From linux subsystem for windows:
gcc -std=c89 -Werror -Wpedantic -Wall -Wno-unused-function -Wno-unused-variable main.c && \
cp main.{c,cc} && \
g++ -std=c++98 -Werror -Wpedantic -Wall -Wno-unused-function -Wno-unused-variable main.cc && \
./a.out
"""
from typing import NamedTuple, Tuple, List, Union, Optional, Callable, Iterable
import abc
import argparse
import contextlib
import os
import re
import sys
import typing

_scriptdir = os.path.dirname(os.path.realpath(__file__))

_special_method_prefixes = [
    'GET',
    'SET',
]

_special_method_names = {
    'Call',
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
}

SYMBOLS = [
    '\n',
    '||', '&&',
    ';', '#', '?', ':', '!',
    '.', ',', '!', '@', '^', '&', '+', '-', '/', '%', '*', '.', '=', '==', '<',
    '>', '<=', '>=', '!=', '(', ')', '{', '}', '[', ']',
]

KEYWORDS = {
    'is', 'not', 'null', 'true', 'false', 'new', 'and', 'or',
    'extern', 'class', 'trait', 'final',
    'for', 'if', 'else', 'while', 'break', 'continue', 'return',
}

PRIMITIVE_TYPES = {
    'void',
    'bool',
    'int',
    'double',
    'function',
    'type',
}

_primitive_method_names = {
    'null': ['Repr', 'Bool'],
    'bool': ['Repr', 'Bool'],
    'int': [
        'Eq', 'Lt',
        'Add', 'Sub', 'Mul', 'Div', 'Mod',
        'Repr', 'Bool',
    ],
    'double': [
        'Eq', 'Lt',
        'Add', 'Sub', 'Mul', 'Div',
        'Repr', 'Bool',
    ],
    'function': ['GETname', 'Repr', 'Bool'],
    'type': [
        'Eq',
        'GETname', 'Repr', 'Bool',
    ],
}

def nullable(type_):
    return type_ not in ('void', 'bool', 'int', 'double')

with open(os.path.join(_scriptdir, 'klprelude.c')) as f:
    CPRELUDE = f.read()

with open(os.path.join(_scriptdir, 'builtins.k')) as f:
    BUILTINS = f.read()


class InverseSet:
    def __init__(self, items):
        self._items = frozenset(items)

    def __contains__(self, key):
        return key not in self._items

    def __repr__(self):
        return f'InverseSet({self._items})'


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
        return f'on line {lineno}\n{line}\n{spaces}*\n'


class Error(Exception):
    def __init__(self, tokens: Iterable[Token], message: str) -> None:
        super().__init__(''.join(token.info for token in tokens) + message)


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


def _make_lexer():
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
        if isinstance(node, GlobalDefinition):
            self._set(node.name, node)
        elif isinstance(node, BaseVariableDefinition):
            self.validate_vardef(node)
            if isinstance(node, VariableDefinition):
                self._check_for_shadow(node)
            self._set(node.name, node)
        else:
            raise Error([node.token], f'FUBAR: Unrecognized node type {node}')

    def validate_vardef(self, vardef):
        if vardef.type not in PRIMITIVE_TYPES and vardef.type != 'var':
            typenode = self.get(vardef.type, [vardef.token])
            if isinstance(typenode, TraitDefinition):
                raise Error(
                    [vardef.token],
                    f'Using trait types variable type not supported')
            if not isinstance(typenode, TypeDefinition):
                raise Error([vardef.token], f'{vardef.type} is not a type')

    def _check_for_shadow(self, node):
        # We need to take care not to shadow VariableDefinitions.
        # If VariableDefinitions are ever shadowed, they will cause
        # incorrect behavior when the function tries to release
        # all local variables before a return.
        scope = self.parent
        name = node.name
        while scope:
            if name in scope.table and isinstance(scope.table[name], VariableDefinition):
                raise Error([node.token, scope.table[name].token],
                            f'Shadowed local variable')
            scope = scope.parent

    def _missing_name_err(self, name, tokens):
        return Error(tokens, f'Name {name} not defined in this scope')

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


@contextlib.contextmanager
def with_frame(ctx, token):
    src = ctx.src
    filename = token.source.filename.replace(os.sep, '/')
    fname = ctx.fctx.fdef.name
    src += f'KLC_push_frame("{filename}", "{fname}", {token.lineno});'
    yield
    src += f'KLC_pop_frame();'


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


class BaseVariableDefinition(Node):
    fields = (
        ('type', str),
        ('name', str),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.type == 'void':
            raise Error([self.token], f'Variable of type void is not allowed')

    def cproto(self, ctx):
        return f'{ctx.cdecltype(self.type)} {ctx.cname(self.name)}'

    def cname(self, ctx):
        return ctx.cname(self.name)

    def crelease(self, ctx):
        return _crelease(ctx, self.type, self.cname(ctx))

    def cretain(self, ctx):
        return _cretain(ctx, self.type, self.cname(ctx))


class Parameter(BaseVariableDefinition):
    pass


class Field(BaseVariableDefinition):
    pass


class GlobalDefinition(Node):
    pass


class Statement(Node):
    def translate(self, ctx):
        """Translates self with the given translation context.
        Returns True if the current statement is terminal (e.g. return).
        No statements directly after a translate that returns True should
        be emitted, since those following statements will be unreachable.
        """
        raise NotImplementedError()


class Expression(Node):
    def translate(self, ctx):
        """Translates self with the given translation context.
        Returns (type, tempvar) pair, where type is the type of
        evaluating this expression, and tempvar is the name of the
        temporary C variable where the result of this expression
        is stored.
        If type is 'void', tempvar should be None.
        """
        raise NotImplementedError()

    def translate_value(self, ctx):
        etype, evar = self.translate(ctx)
        if etype == 'void':
            raise Error([self.token], 'void type not allowed here')
        return etype, evar


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
        return f'KLCN{name}'

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
            return f'({self.cdecltype(type_)}) KLC_var_to_object({cname}, &KLC_type{type_})'


class Name(Expression):
    fields = (
        ('name', str),
    )

    def translate(self, ctx):
        defn = ctx.scope.get(self.name, [self.token])
        if isinstance(defn, GlobalVariableDefinition):
            etype = defn.type
            tempvar = ctx.mktemp(etype)
            ctx.src += f'{tempvar} = KLC_get_global{defn.name}();'
            return (etype, tempvar)
        elif isinstance(defn, BaseVariableDefinition):
            etype = defn.type
            tempvar = ctx.mktemp(etype)
            ctx.src += f'{tempvar} = {ctx.cname(self.name)};'
            ctx.src += _cretain(ctx, etype, tempvar)
            return (etype, tempvar)
        elif isinstance(defn, ClassDefinition):
            tempvar = ctx.mktemp('type')
            ctx.src += f'{tempvar} = &KLC_type{defn.name};'
            return 'type', tempvar
        elif isinstance(defn, FunctionDefinition):
            tempvar = ctx.mktemp('function')
            ctx.src += f'{tempvar} = &KLC_functioninfo{defn.name};'
            return 'function', tempvar
        else:
            raise Error([self.token, defn.token],
                        f'{name} is not a variable')


class SetName(Expression):
    fields = (
        ('name', str),
        ('expression', Expression),
    )

    def translate(self, ctx):
        defn = ctx.scope.get(self.name, [self.token])
        if isinstance(defn, GlobalVariableDefinition):
            raise Error(
                [self.token, defn.token],
                f'Global variables are final ({defn.name})')
        elif isinstance(defn, BaseVariableDefinition):
            etype = defn.type
            tempvar = ctx.mktemp(etype)
            etype, evar = self.expression.translate(ctx)
            ctx.src += _cretain(ctx, etype, evar)
            ctx.src += defn.crelease(ctx)
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

    def translate(self, ctx):
        etype, etempvar = self.expression.translate(ctx)
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


class NullLiteral(Expression):
    fields = (
        ('type', str),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not nullable(self.type):
            raise Error([self.token], f'{self.type} is not nullable')

    def translate(self, ctx):
        tempvar = ctx.mktemp(self.type)
        if self.type == 'var':
            ctx.src += f'{tempvar} = KLC_null;'
        else:
            ctx.src += f'{tempvar} = NULL;'
        return self.type, tempvar


class BoolLiteral(Expression):
    fields = (
        ('value', bool),
    )

    def translate(self, ctx):
        tempvar = ctx.mktemp('bool')
        ctx.src += f'{tempvar} = {1 if self.value else 0};'
        return ('bool', tempvar)


class StringLiteral(Expression):
    fields = (
        ('value', str),
    )

    def translate(self, ctx):
        tempvar = ctx.mktemp('String')
        # TODO: properly escape the string literal
        s = (self.value
            .replace('\\', '\\\\')
            .replace('\t', '\\t')
            .replace('\n', '\\n')
            .replace('"', '\\"')
            .replace("'", "\\'"))
        ctx.src += f'{tempvar} = KLC_mkstr("{s}");'
        return ('String', tempvar)


class IntLiteral(Expression):
    fields = (
        ('value', int),
    )

    def translate(self, ctx):
        tempvar = ctx.mktemp('int')
        # TODO: Warn if size of 'value' is too big
        ctx.src += f'{tempvar} = {self.value}L;'
        return ('int', tempvar)


class DoubleLiteral(Expression):
    fields = (
        ('value', float),
    )

    def translate(self, ctx):
        tempvar = ctx.mktemp('double')
        # TODO: Warn if size of 'value' is too big
        ctx.src += f'{tempvar} = {self.value};'
        return ('double', tempvar)


class ListDisplay(Expression):
    fields = (
        ('expressions', List[Expression]),
    )

    def translate(self, ctx):
        argvars = []
        for expr in self.expressions:
            etype, evar = expr.translate(ctx)
            argvars.append(ctx.varify(etype, evar))
        xvar = ctx.mktemp('List')
        ctx.src += f'{xvar} = KLC_mklist({len(argvars)});'
        for arg in argvars:
            ctx.src += f'KLCNList_mpush({xvar}, {arg});'
        return 'List', xvar


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

    def translate(self, ctx):
        fast_op_types = type(self).fast_op_types
        fast_op = type(self).fast_op
        fallback_method = type(self).fallback_method
        fallback_fn = type(self).fallback_fn
        op_for_null = type(self).op_for_null
        reverse_method_args = type(self).reverse_method_args
        invert_method_result = type(self).invert_method_result

        ltok = self.left.token
        ltype, lvar = self.left.translate_value(ctx)

        rtok = self.right.token
        rtype, rvar = self.right.translate_value(ctx)

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


class Is(BinaryComparison):
    fast_op_types = InverseSet({'var'})
    fast_op = '=='
    fallback_fn = '_Is'


class IsNot(BinaryComparison):
    fast_op_types = InverseSet({'var'})
    fast_op = '!='
    fallback_fn = '_IsNot'


class Equals(BinaryComparison):
    fast_op_types = frozenset(PRIMITIVE_TYPES)
    fast_op = '=='
    op_for_null = '=='
    fallback_method = 'Eq'
    fallback_fn = '_Eq'


class NotEquals(BinaryComparison):
    fast_op_types = frozenset(PRIMITIVE_TYPES)
    fast_op = '!='
    op_for_null = '!='
    fallback_method = 'Eq'
    invert_method_result = True
    fallback_fn = '_Ne'


class LessThan(BinaryComparison):
    fast_op_types = ['int', 'double']
    fast_op = '<'
    fallback_method = 'Lt'
    fallback_fn = '_Lt'


class LessThanOrEqual(BinaryComparison):
    fast_op_types = ['int', 'double']
    fast_op = '<='
    fallback_method = 'Lt'
    invert_method_result = True
    reverse_method_args = True
    fallback_fn = '_Le'


class GreaterThan(BinaryComparison):
    fast_op_types = ['int', 'double']
    fast_op = '>'
    fallback_method = 'Lt'
    reverse_method_args = True
    fallback_fn = '_Gt'


class GreaterThanOrEqual(BinaryComparison):
    fast_op_types = ['int', 'double']
    fast_op = '>='
    fallback_method = 'Lt'
    invert_method_result = True
    fallback_fn = '_Ge'


class FunctionCall(Expression):
    fields = (
        ('function', str),
        ('args', List[Expression]),
    )

    def translate(self, ctx):
        defn = ctx.scope.get(self.function, [self.token])
        if isinstance(defn, FunctionDefinition):
            argtriples = []
            for arg in self.args:
                argtype, argtempvar = arg.translate(ctx)
                argtriples.append((arg.token, argtype, argtempvar))
            return _translate_fcall(ctx, self.token, defn, argtriples)

        if (isinstance(defn, BaseVariableDefinition) and
                defn.type in ('var', 'function')):
            argtempvars = []
            for arg in self.args:
                argtype, argtempvar = arg.translate(ctx)
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

        if isinstance(defn, BaseVariableDefinition):
            # At this point, we know defn.type cannot be var or function
            # For these types, we just want to call the 'Call' method.
            argtriples = [
                (self.token, defn.type, ctx.cname(defn.name)),
            ]
            for arg in self.args:
                argtype, argtempvar = arg.translate_value(ctx)
                argtriples.append((arg.token, argtype, argtempvar))
            return _translate_mcall(ctx, self.token, 'Call', argtriples)

        if isinstance(defn, ClassDefinition):
            argtriples = []
            for arg in self.args:
                argtype, argtempvar = arg.translate(ctx)
                argtriples.append((arg.token, argtype, argtempvar))
            malloc_name = f'KLC_malloc{defn.name}'
            fname = f'{defn.name}_new'
            fdefn = ctx.scope.get(fname, [self.token])
            if not isinstance(fdefn, FunctionDefinition):
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
            fname = f'{ownertype}_m{name}'
        else:
            cdef = ctx.scope.get(ownertype, [token])
            assert isinstance(cdef, ClassDefinition), cdef
            if name not in cdef.method_map(ctx):
                raise _no_such_method_error(token, name, ownertype)
            fname = cdef.method_map(ctx)[name]

        # TODO: Consider looking up from global context
        # to avoid coincidental names that shadow method names
        defn = ctx.scope.get(fname, [token])
        if not isinstance(defn, FunctionDefinition):
            raise Error([token], f'FUBAR: shadowed method {fname}')

        return _translate_fcall(ctx, token, defn, argtriples)

class MethodCall(Expression):
    fields = (
        ('owner', Expression),
        ('name', str),
        ('args', List[Expression]),
    )

    def translate(self, ctx):
        ownertype, ownertempvar = self.owner.translate(ctx)
        argtriples = [(self.token, ownertype, ownertempvar)]
        for arg in self.args:
            argtype, argvar = arg.translate(ctx)
            argtriples.append((arg.token, argtype, argvar))
        return _translate_mcall(ctx, self.token, self.name, argtriples)


class LogicalNot(Expression):
    fields = (
        ('expression', Expression),
    )

    def translate(self, ctx):
        etype, evar = self.expression.translate(ctx)
        xvar = ctx.mktemp('bool')
        ctx.src += f'{xvar} = !{_ctruthy(ctx, etype, evar)};'
        return 'bool', xvar


class LogicalOr(Expression):
    fields = (
        ('left', Expression),
        ('right', Expression),
    )

    def translate(self, ctx):
        ltype, lvar = self.left.translate(ctx)
        xvar = ctx.mktemp('bool')
        ctx.src += f'if (!{_ctruthy(ctx, ltype, lvar)})'
        ctx.src += '{'
        rtype, rvar = self.right.translate(ctx)
        ctx.src += f'{xvar} = {_ctruthy(ctx, rtype, rvar)};'
        ctx.src += '} else {'
        ctx.src += f'{xvar} = 1;'
        ctx.src += '}'
        if 'void' in (ltype, rtype):
            raise Error([self.token], 'void type in or operator')
        return ('bool', xvar)


class LogicalAnd(Expression):
    fields = (
        ('left', Expression),
        ('right', Expression),
    )

    def translate(self, ctx):
        ltype, lvar = self.left.translate(ctx)
        xvar = ctx.mktemp('bool')
        ctx.src += f'if ({_ctruthy(ctx, ltype, lvar)})'
        ctx.src += '{'
        rtype, rvar = self.right.translate(ctx)
        ctx.src += f'{xvar} = {_ctruthy(ctx, rtype, rvar)};'
        ctx.src += '} else {'
        ctx.src += f'{xvar} = 0;'
        ctx.src += '}'
        if 'void' in (ltype, rtype):
            raise Erorr([self.token], 'void type in and operator')
        return ('bool', xvar)


class Conditional(Expression):
    fields = (
        ('condition', Expression),
        ('left', Expression),
        ('right', Expression),
    )

    def translate(self, ctx):
        ctype, cvar = self.condition.translate(ctx)
        ctx.src += f'if ({_ctruthy(ctx, ctype, cvar)})'
        ctx.src += '{'
        ltype, lvar = self.left.translate(ctx)
        lsrc = ctx.src.spawn()
        ctx.src += '} else {'
        rtype, rvar = self.right.translate(ctx)
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

    def translate(self, ctx):
        if self.expression:
            ectx = ctx.ectx()
            etype, tempvar = self.expression.translate(ectx)
            value = tempvar
            if self.type is None:
                self.type = etype
            elif self.type == 'var' and etype != 'void':
                value = ctx.varify(etype, tempvar)
            elif self.type != etype:
                raise Error([self.token],
                            f'Expected {self.type} but got {etype}')
            ectx.src += f'{self.cname(ectx)} = {value};'
            ectx.release_tempvars(tempvar)

        ctx.scope.add(self)


class ExpressionStatement(Statement):
    fields = (
        ('expression', Optional[Expression]),
    )

    def translate(self, ctx):
        ectx = ctx.ectx()
        self.expression.translate(ectx)
        ectx.release_tempvars()


class Return(Statement):
    fields = (
        ('expression', Optional[Expression]),
    )

    def translate(self, ctx):
        ectx = ctx.ectx()
        if self.expression:
            rtype, tempvar = self.expression.translate(ectx)
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


def _release_for_return(ctx, src):
    # Before returning, we should release all local variables
    # We need to take care not to release function parameters
    # or global variables
    scope = ctx.scope
    while scope:
        for vdef in reversed(scope.local_definitions):
            if isinstance(vdef, VariableDefinition):
                src += vdef.crelease(ctx)
        scope = scope.parent


class Block(Statement):
    fields = (
        ('statements', List[Statement]),
    )

    def translate(self, pctx):
        pctx.src += '{'
        prologue = pctx.src.spawn(1)
        ctx = pctx.bctx(1)
        epilogue = pctx.src.spawn(1)
        pctx.src += '}'
        early_return = False
        for i, statement in enumerate(self.statements):
            if statement.translate(ctx):
                early_return = True
                if i + 1 < len(self.statements):
                    raise Error([self.statements[i + 1].token],
                                'Unreachable statement')
                break

        # Declare the local variables for C
        # To be C89 compatible, we need all variable definitions
        # to appear at the beginning of the block.
        for vdef in ctx.scope.local_definitions:
            assert isinstance(vdef, VariableDefinition), vdef
            prologue += f'{vdef.cproto(ctx)} = {ctx.czero(vdef.type)};'

        # If there's an early return, there's no need to have an
        # epilogue and generate unreachable code.
        if not early_return:
            # If we don't have an early return, we should
            # make sure to release all local variables defined in
            # this block before exiting
            # We should also release in LIFO order
            for vdef in reversed(ctx.scope.local_definitions):
                if vdef.type not in PRIMITIVE_TYPES:
                    epilogue += vdef.crelease(ctx)


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

    def translate(self, ctx):
        ctx.src += '{'
        ctx.src += 'KLC_bool b;'
        ectx = ctx.ectx()
        rtype, tempvar = self.condition.translate(ectx)
        ectx.src += f'b = {_ctruthy(ctx, rtype, tempvar)};'
        ectx.release_tempvars()
        ctx.src += 'if (b)'
        self.body.translate(ctx)
        if self.other is not None:
            ctx.src += 'else'
            self.other.translate(ctx)
        ctx.src += '}'


class While(Statement):
    fields = (
        ('condition', Expression),
        ('body', Block),
    )

    def translate(self, ctx):
        ctx.src += 'while (1) {'
        ctx.src += 'KLC_bool b;'
        ectx = ctx.ectx()
        rtype, tempvar = self.condition.translate(ectx)
        ectx.src += f'b = {_ctruthy(ctx, rtype, tempvar)};'
        ectx.release_tempvars()
        ctx.src += 'if (!b) { break; }'
        self.body.translate(ctx)
        ctx.src += '}'


class GlobalTranslationContext(TranslationContext):
    def __init__(self, program):
        super().__init__(Scope(None))
        self.gctx = self
        for d in program.definitions:
            self.scope.add(d)
        self.out = FractalStringBuilder(0)
        self.out(CPRELUDE)
        self.fwd = self.out.spawn()
        self.hdr = self.out.spawn()
        self.src = self.out.spawn()

    def fctx(self, fdef):
        "Create child function translation context"
        return FunctionTranslationContext(self, fdef)


class BodyTranslationContext(TranslationContext):

    def bctx(self, depth):
        return BlockTranslationContext(self, depth)


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


class BlockTranslationContext(BodyTranslationContext):
    def __init__(self, parent: BodyTranslationContext, depth):
        super().__init__(Scope(parent.scope))
        self.parent = parent
        self.fctx = parent.fctx
        self.gctx = parent.gctx
        self.src = parent.src.spawn(depth)

    def ectx(self):
        return ExpressionTranslationContext(self)


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


class Program(Node):
    fields = (
        ('definitions', List[GlobalDefinition]),
    )

    def translate(self):
        ctx = GlobalTranslationContext(self)
        for ptype, mnames in sorted(_primitive_method_names.items()):
            methodmap = _compute_method_map(
                token=self.token,
                cname=ptype,
                method_names=mnames,
                trait_names=['Object'],
                ctx=ctx)
            _write_ctypeinfo(ctx.src, ptype, methodmap, use_null_deleter=True)
        for d in self.definitions:
            d.translate(ctx)
        return str(ctx.out)


class GlobalVariableDefinition(GlobalDefinition):
    fields = (
        ('extern', bool),
        ('type', str),
        ('name', str),
    )

    def translate(self, ctx):
        ctype = ctx.cdecltype(self.type)
        ctx.hdr += f'{ctype} KLC_get_global{self.name}();'
        ctx.src += f'int KLC_initialized_global{self.name} = 0;'
        ctx.src += f'{ctype} KLC_global{self.name} = {ctx.czero(self.type)};'

        ctx.src += f'{ctype} KLC_get_global{self.name}() ' '{'
        ctx.src += f'  if (!KLC_initialized_global{self.name}) ' '{'
        ctx.src += f'    KLC_global{self.name} = KLCN_init{self.name}();'
        ctx.src += f'    KLC_initialized_global{self.name} = 1;'
        if self.type in PRIMITIVE_TYPES:
            pass
        elif self.type == 'var':
            ctx.src += f'    KLC_release_var_on_exit(KLC_global{self.name});'
        else:
            ctx.src += f'    KLC_release_object_on_exit((KLC_header*) KLC_global{self.name});'
        ctx.src += '  }'
        src1 = ctx.src.spawn(1)
        src1 += _cretain(ctx, self.type, f'KLC_global{self.name}')
        src1 += f'return KLC_global{self.name};'
        ctx.src += '}'


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
            if self.name.endswith(f'_m{m}'):
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

    def cproto(self, ctx):
        crt = ctx.cdecltype(self.return_type)
        cname = ctx.cname(self.name)
        cparams = ', '.join(p.cproto(ctx) for p in self.params)
        return f'{crt} {cname}({cparams})'

    def untyped_cproto(self, ctx):
        return f'KLC_var KLC_untyped{self.name}(int argc, KLC_var* argv)'

    def translate(self, gctx: GlobalTranslationContext):
        ctx = gctx.fctx(self)

        rt = self.return_type
        if rt not in PRIMITIVE_TYPES and rt != 'var':
            rtnode = ctx.scope.get(self.return_type, [self.token])
            if isinstance(rtnode, TraitDefinition):
                raise Error(
                    [vardef.token],
                    f'Declaring trait as return type not supported')
            if not isinstance(rtnode, TypeDefinition):
                raise Error([vardef.token], f'{vardef.type} is not a type')

        self._translate_untyped(ctx)

        ctx.hdr += self.cproto(ctx) + ';'
        if self.body:
            ctx.src += self.cproto(ctx)
            self.body.translate(ctx)

    def _translate_untyped(self, ctx):
        name = self.name
        ctx.hdr += self.untyped_cproto(ctx) + ';'
        ctx.hdr += f'KLC_functioninfo KLC_functioninfo{name} = ' '{'
        ctx.hdr += f'  "{name}",'
        ctx.hdr += f'  KLC_untyped{self.name},'
        ctx.hdr += '};'
        ctx.src += self.untyped_cproto(ctx) + '{'
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


class TypeDefinition(GlobalDefinition):
    pass


def _delname(cname):
    return f'KLC_delete{cname}'


def _write_ctypeinfo(src, cname, methodmap, use_null_deleter=False):
    # For primitive types, it's silly to have a deleter, so
    # use_null_deleter allows caller to control this
    del_name = _delname(cname)
    if methodmap:
        src += f'static KLC_methodinfo KLC_methodarray{cname}[] = ' '{'
        for mname, mfname in sorted(methodmap.items()):
            src += '  {' f'"{mname}", KLC_untyped{mfname}' '},'
        src += '};'

    src += f'static KLC_methodlist KLC_methodlist{cname} = ' '{'
    src += f'  {len(methodmap)},'
    src += f'  KLC_methodarray{cname},' if methodmap else '  NULL,'
    src += '};'

    src += f'KLC_typeinfo KLC_type{cname} = ' '{'
    src += f'  "{cname}",'
    src += '  NULL,' if use_null_deleter else f'  &{del_name},'
    src += f'  &KLC_methodlist{cname},'
    src += '};'


def _check_type_name(token, name):
    if '_' in name:
        raise Error([token], 'Class and trait names cannot have underscores')
    if name[:1].lower() == name[:1]:
        raise Error([token],
                    'Class and trait names must start with uppercase letter')


def _check_all_are_traits(token, traits, ctx):
    for trait_name in traits:
        trait_defn = ctx.scope.get(trait_name, [token])
        if not isinstance(trait_defn, TraitDefinition):
            raise Error([token, trait_defn.token],
                        f'{trait_name} is not a trait')


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
        _check_type_name(self.token, self.name)

    def translate(self, ctx: GlobalTranslationContext):
        # Verify that there's no circular trait inheritance,
        # and in the process, also verify all entries listed in 'traits'
        # are actually defined and are traits.
        self.trait_method_map(ctx)

    def trait_method_map(self, ctx, stack=None):
        if self._trait_method_map is None:
            self._trait_method_map = _compute_method_map(
                token=self.token,
                cname=self.name,
                method_names=self.methods,
                trait_names=self.traits,
                ctx=ctx,
                stack=stack)

        return self._trait_method_map


def _compute_method_map(token, cname, method_names, trait_names, ctx, stack=None):
    ctx = ctx.gctx  # double check that this is the global context
    stack = [] if stack is None else stack
    if cname in stack:
        raise Error([tdef.token for tdef in stack],
                    f'Circular trait inheritance')
    _check_all_are_traits(token, trait_names, ctx)
    method_map = {mname: f'{cname}_m{mname}' for mname in method_names}
    traits = [ctx.scope.get(n, [token]) for n in trait_names]
    stack.append(cname)
    # MRO is DFS
    for trait in traits:
        for mname, mfname in trait.trait_method_map(ctx, stack).items():
            if mname not in method_map:
                method_map[mname] = mfname
    stack.pop()
    return method_map


class ClassDefinition(TypeDefinition):
    fields = (
        ('name', str),
        ('traits', List[str]),
        ('fields', Optional[List[Field]]),
        ('methods', List[str]),
    )

    _method_map = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _check_type_name(self.token, self.name)

    @property
    def extern(self):
        return self.fields is None

    def method_map(self, ctx):
        if self._method_map is None:
            self._method_map = _compute_method_map(
                token=self.token,
                cname=self.name,
                method_names=self.methods,
                trait_names=self.traits,
                ctx=ctx)
        return self._method_map

    def translate(self, ctx: GlobalTranslationContext):
        _check_all_are_traits(self.token, self.traits, ctx)
        name = self.name
        cname = ctx.cname(name)
        cdecltype = ctx.cdecltype(name)

        del_name = _delname(name)
        malloc_name = f'KLC_malloc{name}'

        delete_proto = f'void {del_name}(KLC_header* robj, KLC_header** dq)'
        malloc_proto = f'{cdecltype} {malloc_name}()'

        ctx.hdr += delete_proto + ';'
        ctx.hdr += malloc_proto + ';'

        ctx.hdr += f'extern KLC_typeinfo KLC_type{name};'

        _write_ctypeinfo(
            src=ctx.src,
            cname=name,
            methodmap=self.method_map(ctx))

        if self.extern:
            return

        self._translate_field_implementations(ctx)

        # if extern, this typedef should already exist
        ctx.fwd += f'typedef struct {cname} {cname};'

        ctx.hdr += f'struct {cname} ' '{'
        ctx.hdr += '  KLC_header header;'
        for field in self.fields:
            ctx.hdr += f'  {field.cproto(ctx)};'
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
        ctx.src += f'  KLC_init_header(&obj->header, &KLC_type{name});'
        for field in self.fields:
            cfname = ctx.cname(field.name)
            ctx.src += f'  obj->{cfname} = {ctx.czero(field.type)};'
        ctx.src += '  return obj;'
        ctx.src += '}'

    def _translate_field_implementations(self, ctx):
        this_proto = f"{ctx.cdecltype(self.name)} {ctx.cname('this')}"
        for field in self.fields:
            field_ref = f'KLCNthis->{ctx.cname(field.name)}'
            ctype = ctx.cdecltype(field.type)

            ## GETTER
            getter_name = f'{self.name}_mGET{field.name}'
            getter_cname = ctx.cname(getter_name)
            getter_proto = f'{ctype} {getter_cname}({this_proto})'
            ctx.src += getter_proto + '{'
            sp = ctx.src.spawn(1)
            ctx.src += '}'
            sp += _cretain(ctx, field.type, f'({field_ref})')
            sp += f'return {field_ref};'

            # SETTER
            setter_name = f'{self.name}_mSET{field.name}'
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


def parse(source):
    cache = dict()
    stack = []
    main_node = parse_one_source(source, cache, stack)
    defs = list(main_node.definitions)
    for lib_node in cache.values():
        defs.extend(lib_node.definitions)
    return Program(main_node.token, defs)


def _has_lower(s):
    return s.upper() != s

def _has_upper(s):
    return s.lower() != s

def _is_special_method_name(name):
    return (name in _special_method_names or
            any(name.startswith(p) for p in _special_method_prefixes))

def _check_method_name(token, name):
    if _has_upper(name[0]) and not _is_special_method_name(name):
        raise Error(
            [token],
            f'Only special methods may start with an upper case letter')


def parse_one_source(source, cache, stack):
    tokens = lex(source)
    i = 0
    cache = dict() if cache is None else cache
    stack = [] if stack is None else stack
    indent_stack = []

    def peek(j=0):
        nonlocal i
        if should_skip_newlines():
            while i < len(tokens) and tokens[i] == '\n':
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
        if token.type == 'NAME' and '__' in token.value:
            raise Error(
                [token],
                'Double underscores are not allowed in identifiers')
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
        token = expect('NAME')
        if token.value != name:
            raise Error([token], f'Expected name {name} but got {token.value}')
        return token

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

    def parse_program():
        consume_delim()
        token = peek()
        defs = []
        while not at('EOF'):
            parse_global_definition(defs)
        return Program(token, defs)

    def parse_global_definition(defs):
        if at('#'):
            parse_macro(defs)
            return

        if at('trait'):
            parse_trait_definition(defs)
            return

        if at('class') or at('extern') and at('class', 1):
            parse_class_definition(defs)
            return

        if at('NAME') and at('NAME', 1) and at('(', 2):
            parse_function_definition(defs)
            return

        if (at('extern') or at('NAME')) and at('NAME', 1):
            parse_global_variable_definition(defs)
            return

        raise Error([peek()], f'Expected class, function or variable definition')

    def parse_macro(defs):
        token = expect('#')
        expect_name('include')
        upath = expect('STRING').value
        consume_delim()
        if not upath.startswith('./'):
            raise Error([token], 'Absolute paths not supported yet')

        if upath in [p for p, _ in stack]:
            toks = [t for _, t in stack]
            raise Error(toks + [token], '#include cycle')

        if upath in cache:
            return

        parts = upath.split('/')
        assert parts[0] == '.'
        dirpath = os.path.dirname(source.filename)
        path = os.path.join(dirpath, *parts[1:])
        abspath = os.path.abspath(os.path.realpath(path))
        if not os.path.isfile(abspath):
            raise Error([token], f'File {upath} ({abspath}) does not exist')

        with open(abspath) as f:
            data = f.read()

        try:
            stack.append((upath, token))
            cache[upath] = parse_one_source(Source(abspath, data), cache, stack)
        finally:
            stack.pop()

    def parse_global_variable_definition(defs):
        token = peek()
        extern = bool(consume('extern'))
        vtype = expect('NAME').value
        vname = expect('NAME').value
        defs.append(GlobalVariableDefinition(token, extern, vtype, vname))
        ifname = f'_init{vname}'
        if extern:
            initf = FunctionDefinition(
                token,
                vtype,
                ifname,
                [],
                None)
        else:
            if consume('='):
                expr = parse_expression()
                initf = FunctionDefinition(
                    token,
                    vtype,
                    ifname,
                    [],
                    Block(token, [Return(token, expr)]))
            else:
                initf = FunctionDefinition(
                    token,
                    vtype,
                    ifname,
                    [],
                    Block(token, [
                        VariableDefinition(token, True, vtype, 'ret', None),
                        Return(token, Name(token, 'ret')),
                    ]))
        defs.append(initf)
        expect_delim()

    def parse_function_definition(defs):
        token = peek()
        return_type = expect('NAME').value
        nametoken = peek()
        name = expect('NAME').value
        if name[:1].lower() != name[:1]:
            raise Error(
                [nametoken],
                f'Function names should not start with uppercase letters')
        params = parse_params()
        if at('{'):
            body = parse_block()
        else:
            body = None
            expect_delim()
        defs.append(FunctionDefinition(token, return_type, name, params, body))

    def parse_class_definition(defs):
        token = peek()
        extern = bool(consume('extern'))
        expect('class')
        name = expect('NAME').value
        traits = parse_trait_list()
        method_to_token_table = dict()
        fields = None if extern else []
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
                fname = f'{name}_new'
                mtoken = expect('new')
                declparams = parse_params()
                if extern:
                    # For extern types, the 'new' function should return
                    # the constructed object. Further, the new function
                    # itself must be extern
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
                    body = parse_block()
                    params = [Parameter(mtoken, name, 'this')] + declparams
                newdef = FunctionDefinition(mtoken, rt, fname, params, body)
                defs.append(newdef)
            elif at('NAME') and at('NAME', 1) and at_delim(2):
                if extern:
                    raise Error(
                        [peek()],
                        'Extern classes cannot declare fields '
                        f'(in definition of class {name})')
                ftoken = peek()
                ftype = expect('NAME').value
                fname = expect('NAME').value
                expect_delim()
                fields.append(Field(ftoken, ftype, fname))

                # GET and SET methods are implemented specially
                # during the class translation
                getter_name = f'{name}_mGET{fname}'
                setter_name = f'{name}_mSET{fname}'
                defs.append(FunctionDefinition(
                    ftoken,
                    ftype,
                    getter_name,
                    [Parameter(ftoken, name, 'this')],
                    None))
                defs.append(FunctionDefinition(
                    ftoken,
                    ftype,
                    setter_name,
                    [Parameter(ftoken, name, 'this'),
                     Parameter(ftoken, ftype, 'value')],
                    None))
                method_to_token_table[f'GET{fname}'] = ftoken
                method_to_token_table[f'SET{fname}'] = ftoken

            else:
                mtoken = peek()
                rtype = expect('NAME').value
                mname = expect('NAME').value
                _check_method_name(mtoken, mname)
                params = [Parameter(mtoken, name, 'this')] + parse_params()
                body = None if consume_delim() else parse_block()

                # A method is mapped to a function with a special name,
                # and an implicit first parameter.
                fname = f'{name}_m{mname}'

                if mname in method_to_token_table:
                    raise Error([method_to_token_table[mname], mtoken],
                                f'Duplicate method {name}.{mname}')

                method_to_token_table[mname] = mtoken

                defs.append(FunctionDefinition(mtoken, rtype, fname, params, body))

        consume_delim()

        if not extern and newdef is None:
            defs.append(FunctionDefinition(
                token,
                'void',
                f'{name}_new',
                [Parameter(mtoken, name, 'this')],
                Block(token, [])))

        method_names = sorted(method_to_token_table)
        defs.append(ClassDefinition(token, name, traits, fields, method_names))

    def parse_trait_definition(defs):
        token = expect('trait')
        name = expect('NAME').value
        method_to_token_table = dict()
        traits = parse_trait_list()
        expect('{')
        consume_delim()
        while not consume('}'):
            mtoken = peek()
            rtype = expect('NAME').value
            mname = expect('NAME').value
            params = parse_params()
            body = parse_block()
            _check_method_name(mtoken, mname)

            # A method is mapped to a function with a special name,
            # and an implicit first parameter.
            fname = f'{name}_m{mname}'
            params = [Parameter(mtoken, 'var', 'this')] + params

            if mname in method_to_token_table:
                raise Error([method_to_token_table[mname], mtoken],
                            f'Duplicate method {name}.{mname}')

            method_to_token_table[mname] = mtoken

            defs.append(FunctionDefinition(mtoken, rtype, fname, params, body))
        consume_delim()

        method_names = sorted(method_to_token_table)
        defs.append(TraitDefinition(token, name, traits, method_names))

    def parse_trait_list():
        if consume('('):
            traits = []
            while not consume(')'):
                traits.append(expect('NAME').value)
                if not consume(','):
                    expect(')')
                    break
        else:
            traits = ['Object']
        return traits

    def parse_block():
        token = expect('{')
        consume_delim()
        statements = []
        while not consume('}'):
            statements.append(parse_statement())
        consume_delim()
        return Block(token, statements)

    def parse_statement():
        token = peek()

        if at('{'):
            return parse_block()

        if (at('final') or at('NAME')) and at('NAME', 1):
            return parse_variable_definition()

        if consume('while'):
            expect('(')
            with skipping_newlines(True):
                condition = parse_expression()
                expect(')')
            body = parse_block()
            return While(token, condition, body)

        if consume('if'):
            expect('(')
            with skipping_newlines(True):
                condition = parse_expression()
                expect(')')
            body = parse_block()
            if consume('else'):
                other = parse_statement()
            else:
                other = None
            return If(token, condition, body, other)

        if consume('return'):
            expression = None if at_delim() else parse_expression()
            expect_delim()
            return Return(token, expression)

        expression = parse_expression()
        expect_delim()
        return ExpressionStatement(token, expression)

    def parse_expression():
        return parse_conditional()

    def parse_conditional():
        expr = parse_or()
        token = peek()
        if consume('?'):
            left = parse_expression()
            expect(':')
            right = parse_conditional()
            return Conditional(token, expr, left, right)
        return expr

    def parse_or():
        expr = parse_and()
        while True:
            token = peek()
            if consume('or'):
                right = parse_and()
                expr = LogicalOr(token, expr, right)
            else:
                break
        return expr

    def parse_and():
        expr = parse_relational()
        while True:
            token = peek()
            if consume('and'):
                right = parse_relational()
                expr = LogicalAnd(token, expr, right)
            else:
                break
        return expr

    def parse_relational():
        expr = parse_additive()
        while True:
            token = peek()
            if consume('=='):
                expr = Equals(token, expr, parse_additive())
            elif consume('!='):
                expr = NotEquals(token, expr, parse_additive())
            elif consume('<'):
                expr = LessThan(token, expr, parse_additive())
            elif consume('<='):
                expr = LessThanOrEqual(token, expr, parse_additive())
            elif consume('>'):
                expr = GreaterThan(token, expr, parse_additive())
            elif consume('>='):
                expr = GreaterThanOrEqual(token, expr, parse_additive())
            elif consume('is'):
                if consume('not'):
                    expr = IsNot(token, expr, parse_additive())
                else:
                    expr = Is(token, expr, parse_additive())
            else:
                break
        return expr

    def parse_additive():
        expr = parse_multiplicative()
        while True:
            token = peek()
            if consume('+'):
                expr = MethodCall(token, expr, 'Add', [parse_multiplicative()])
            elif consume('-'):
                expr = MethodCall(token, expr, 'Sub', [parse_multiplicative()])
            else:
                break
        return expr

    def parse_multiplicative():
        expr = parse_unary()
        while True:
            token = peek()
            if consume('*'):
                expr = MethodCall(token, expr, 'Mul', [parse_unary()])
            elif consume('/'):
                expr = MethodCall(token, expr, 'Div', [parse_unary()])
            elif consume('%'):
                expr = MethodCall(token, expr, 'Mod', [parse_unary()])
            else:
                break
        return expr

    def parse_unary():
        token = peek()
        if consume('-'):
            expr = parse_pow()
            if isinstance(expr, (IntLiteral, DoubleLiteral)):
                type_ = type(expr)
                return type_(expr.token, -expr.value)
            else:
                return MethodCall(token, expr, 'Neg', [])
        if consume('!'):
            expr = parse_pow()
            return LogicalNot(token, expr)
        return parse_pow()

    def parse_pow():
        expr = parse_postfix()
        token = peek()
        if consume('**'):
            expr = MethodCall(token, expr, 'Pow', [parse_pow()])
        return expr

    def parse_postfix():
        expr = parse_primary()
        while True:
            token = peek()
            if consume('.'):
                if consume('('):
                    cast_type = expect('NAME').value
                    expect(')')
                    expr = Cast(token, expr, cast_type)
                    continue
                else:
                    name = expect('NAME').value
                    if at('('):
                        args = parse_args()
                        expr = MethodCall(token, expr, name, args)
                        continue
                    elif consume('='):
                        val = parse_expression()
                        expr = MethodCall(token, expr, f'SET{name}', [val])
                        continue
                    else:
                        expr = MethodCall(token, expr, f'GET{name}', [])
                        continue
            elif at('['):
                args = parse_args('[', ']')
                if consume('='):
                    args.append(parse_expression())
                    expr = MethodCall(token, expr, 'SetItem', args)
                else:
                    expr = MethodCall(token, expr, 'GetItem', args)
                continue
            break
        return expr

    def parse_primary():
        token = peek()

        if consume('('):
            expr = parse_expression()
            expect(')')
            return expr

        if consume('['):
            exprs = []
            while not consume(']'):
                exprs.append(parse_expression())
                if not consume(','):
                    expect(']')
                    break
            return ListDisplay(token, exprs)

        if consume('null'):
            type_ = 'var'
            if consume('('):
                type_ = expect('NAME').value
                expect(')')
            return NullLiteral(token, type_)

        if consume('true'):
            return BoolLiteral(token, True)

        if consume('false'):
            return BoolLiteral(token, False)

        if at('INT'):
            value = expect('INT').value
            return IntLiteral(token, value)

        if at('FLOAT'):
            value = expect('FLOAT').value
            return DoubleLiteral(token, value)

        if at('NAME'):
            name = expect('NAME').value
            if consume('='):
                expr = parse_expression()
                return SetName(token, name, expr)
            elif at('('):
                args = parse_args()
                return FunctionCall(token, name, args)
            else:
                return Name(token, name)

        if at('STRING'):
            return StringLiteral(token, expect('STRING').value)

        raise Error([token], 'Expected expression')

    def parse_variable_definition():
        token = peek()
        final = bool(consume('final'))
        vartype = None if final else expect('NAME').value
        name = expect('NAME').value
        value = parse_expression() if consume('=') else None
        expect_delim()
        if final and value is None:
            raise Error(
                [self.token],
                'final variables definitions must specify an expression')
        return VariableDefinition(token, final, vartype, name, value)

    def parse_args(opener='(', closer=')'):
        args = []
        expect(opener)
        while not consume(closer):
            args.append(parse_expression())
            if not consume(','):
                expect(closer)
                break
        return args

    def parse_params():
        params = []
        expect('(')
        while not consume(')'):
            paramtoken = peek()
            paramtype = expect('NAME').value
            paramname = expect('NAME').value
            params.append(Parameter(paramtoken, paramtype, paramname))
            if not consume(','):
                expect(')')
                break
        return params

    return parse_program()


tok = lex(Source('<dummy>', 'dummy'))[0]

builtins_node = parse(Source('<builtin>', BUILTINS))

parser = argparse.ArgumentParser()
parser.add_argument('kfile')

def main():
    args = parser.parse_args()
    with open(args.kfile) as f:
        data = f.read()
    source = Source(args.kfile, data)
    node = parse(source)
    program = Program(node.token, node.definitions + builtins_node.definitions)
    print(program.translate())

if __name__ == '__main__':
    main()
