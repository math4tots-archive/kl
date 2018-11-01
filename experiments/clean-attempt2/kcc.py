"""kcc = Kyumin C Compiler
"""

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


@Namespace
def U(ns):
    "Utilities"

    @ns
    class CaseClass(object):
        # fields/prefix_fields ->
        # list of (name, type) pairs
        prefix_fields = ()

        def __init__(self, *args):
            tp = type(self)
            name = tp.__name__
            fields = tuple(tp.prefix_fields) + tuple(tp.fields)
            if len(fields) != len(args):
                raise TypeError(
                    f'Expected {len(fields)} args but got {len(args)} args')
            for (n, t), arg in zip(fields, args):
                if not isinstance(arg, t):
                    raise TypeError(
                        f'Expected {name}.{n} to be {t} but got {arg}')
                setattr(self, n, arg)


        def __eq__(self, other):
            if type(self) != type(other):
                return False
            for n, _ in type(self).fields:
                if getattr(self, n) != getattr(other, n):
                    return False
            return True

        def __hash__(self):
            return hash(tuple(getattr(self, n) for n, _ in type(self).fields))

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
    ns(Set, 'Set')

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
    ns(Map, 'Map')

    @ns
    class FractalStringBuilder(object):
        def __init__(self, depth=0):
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


class Source(U.CaseClass):
    fields = (
        ('filename', str),
        ('data', str),
    )


class Token(U.CaseClass):
    fields = (
        ('type', str),
        ('value', object),
        ('source', U.Optional[Source]),
        ('i', U.Optional[int]),
    )


class Node(U.CaseClass):
    prefix_fields = (
        ('token', Token),
    )


@Namespace
def C(ns):

    class N(Node):
        pass

    @ns
    class GlobalDefinition(N):
        pass

    @ns
    class Statement(N):
        pass

    @ns
    class Block(Statement):
        fields = (
            ('statements', U.List[Statement]),
        )

    @ns
    class Expression(N):
        pass

    @ns
    class TranslationUnit(N):
        fields = (
            ('name', str),
            ('definitions', GlobalDefinition),
        )

    @ns
    class BaseInclude(GlobalDefinition):
        fields = (
            ('filename', str),
        )

    @ns
    class BaseVariableDefinition(N):
        fields = (
            ('type', str),
            ('name', str),
        )

    @ns
    class IncludeAngle(BaseInclude):
        pass

    @ns
    class IncludeQuote(BaseInclude):
        pass

    @ns
    class GlobalVariableDefinition(BaseVariableDefinition):
        fields = (
            ('type', str),
            ('name', str),
        )

    @ns
    class Field(BaseVariableDefinition):
        pass

    @ns
    class Struct(GlobalDefinition):
        fields = (
            ('fields', U.List[Field]),
        )

    @ns
    class Parameter(BaseVariableDefinition):
        pass

    @ns
    class Function(GlobalDefinition):
        fields = (
            ('return_type', str),
            ('name', str),
            ('parameters', U.List[Parameter]),
            ('body', Block),
        )

    @ns
    class ExpressionStatement(Statement):
        fields = (
            ('expression', Expression),
        )

    @ns
    class ReturnStatement(Statement):
        fields = (
            ('expression', Expression),
        )

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
    class Int(Expression):
        fields = (
            ('value', int),
        )

    def translate_header(tu: TranslationUnit):
        sb = FractalStringBuilder()


print(U)
