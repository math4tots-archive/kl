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


@Namespace
def util(ns):
    ns += typeutil
    ns += lexutil


@Namespace
def ast(ns):
    Error = util.Error

    @ns
    class Node(object):
        def __init__(self, token, ctx, *args):
            self.token = token
            self.ctx = ctx
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
    class GlobalDefinition(Node):
        pass

    @ns
    class BaseVariableDefinition(Node):
        fields = (
            ('type', str),
            ('name', str),
        )

    @ns
    class Parameter(BaseVariableDefinition):
        pass

    @ns
    class Statement(Node):
        pass

    @ns
    class Expression(Node):
        pass

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
            if self.name == 'main':
                if self.return_type != 'void' or len(self.params):
                    raise Error([self.token],
                                'main function must have signature '
                                "'void main()'")

    @ns
    class ClassDefinition(GlobalDefinition):
        pass



