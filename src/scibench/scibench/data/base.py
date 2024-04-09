from sympy import Symbol
from collections import OrderedDict

EQUATION_CLASS_DICT = OrderedDict()


def register_eq_class(cls):
    EQUATION_CLASS_DICT[cls.__name__] = cls
    return cls


def get_eq_obj(key):
    if key in EQUATION_CLASS_DICT:
        return EQUATION_CLASS_DICT[key]
    raise KeyError(f'`{key}` is not expected as a equation object key')


class KnownEquation(object):
    _eq_name = None
    _operator_set = ['add', 'sub', 'mul', 'div', 'inv', 'pow', 'sin', 'cos', 'exp', 'log', 'const']
    simulated_exec = False

    def __init__(self, num_vars):
        self.num_vars = num_vars
        self.dim = []
        self.sympy_eq = None
        self.x = [Symbol(f'X{i}',real=True) for i in range(num_vars)]