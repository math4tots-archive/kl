import typing


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


@Namespace
def IR(ns):

    class NodeMeta(type):
        def __new__(mcs, name, bases, d):
            fields = dict()

            def _add_field(fname, type_):
                if fname in fields:
                    raise TypeError(f'{name}.{fname}')
                fields[fname] = type_

            for base in bases:
                for fname, type_ in getattr(base, '@fields').items():
                    _add_field(fname, type_)

            if '__annotations__' in d:
                for fname, type_ in d['__annotations__'].items():
                    _add_field(fname, type_)

            d['@fields'] = fields

            defaults = {n: d.get(n, None) for n in fields}

            defaultsname = 'defaults'
            i = 0
            while defaultsname in fields:
                defaultsname = f'defaults{i}'
                i += 1

            selfname = 'self'
            i = 0
            while selfname in fields:
                selfname = f'self{i}'
                i += 1

            if '__init__' not in d:
                argstr = ','.join([selfname] + [f'{n}=None' for n in fields])
                defstr = f'def __init__({argstr}):\n'
                assignstrs = []
                for fname in fields:
                    if defaults[fname] is not None:
                        defaults[fname]()  # verify it's a function
                        assignstrs.append(f'  if {fname} is None:\n')
                        assignstrs.append(
                            f'    {fname} = {selfname}.{defaultsname}'
                            f'["{fname}"]()\n')
                    assignstrs.append(f'  {selfname}.{fname} = {fname}\n')
                if not assignstrs:
                    assignstrs.append(f'  pass\n')
                evalstr = defstr + ''.join(assignstrs)
                localdict = dict()
                global_copy = dict(globals())
                global_copy[defaultsname] = defaults
                exec(evalstr, globals(), localdict)
                d['__init__'] = localdict['__init__']

            cls = super().__new__(mcs, name, bases, d)
            setattr(cls, defaultsname, defaults)

            return cls

    class Node(NodeMeta('Node', (), dict())):
        def __repr__(self):
            fields = getattr(type(self), '@fields')
            argstr = ', '.join(
                f'{n}={repr(getattr(self, n))}' for n in fields)
            return f'{type(self).__name__}({argstr})'

    class Type(Node):
        pass

    class BuiltinType(Type):
        name: str

    VOID = BuiltinType('void')
    BOOL = BuiltinType('bool')
    INT = BuiltinType('int')
    FLOAT = BuiltinType('float')
    STRING = BuiltinType('String')
    LIST = BuiltinType('List')
    MAP = BuiltinType('Map')
    FUNCTION = BuiltinType('Function')

    class Expression(Node):
        pass

    class Parameter(Node):
        type: Type
        name: str

    class Function(Node):
        return_type: Type
        name: str
        parameters: typing.List[Parameter] = lambda: []
        body: Expression
