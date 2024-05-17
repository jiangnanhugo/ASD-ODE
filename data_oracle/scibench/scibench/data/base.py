from sympy import Symbol
from collections import OrderedDict

EQUATION_CLASS_DICT = OrderedDict()
import json

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


    def __init__(self, num_vars, vars_range_and_types):
        self.num_vars = num_vars
        self.dim = []
        self.sympy_eq = None
        self.vars_range_and_types = vars_range_and_types
        self.x = [Symbol(f'X{i}', real=True) for i in range(num_vars)]

    def vars_range_and_types_to_json_str(self):
        if self.vars_range_and_types:
            return json.dumps([one.to_dict() for one in self.vars_range_and_types])
        else:
            default = {'name': 'LogUniform',
                       'range': [0.1, 10],
                       'dim': [1, ],
                       'only_positive': True}
            return json.dumps([default for _ in range(self.num_vars)])