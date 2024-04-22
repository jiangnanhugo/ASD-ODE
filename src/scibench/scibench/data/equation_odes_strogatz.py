import sympy

from collections import OrderedDict
from scibench.data.base import KnownEquation
from scibench.symbolic_data_generator import LogUniformSampling, UniformSampling

EQUATION_CLASS_DICT = OrderedDict()


def register_eq_class(cls):
    EQUATION_CLASS_DICT[cls.__name__] = cls
    return cls


def get_eq_obj(key, **kwargs):
    if key in EQUATION_CLASS_DICT:
        return EQUATION_CLASS_DICT[key](**kwargs)
    raise KeyError(f'`{key}` is not expected as a equation object key')


@register_eq_class
class RCcircuit(KnownEquation):
    _eq_name = 'vars1_prog1'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        c = [0.7, 1.2, 2.31]

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x

        self.sympy_eq = [
            (c[0] - x[0] / c[1]) / c[2]
        ]


@register_eq_class
class PopulationGrowth(KnownEquation):
    _eq_name = 'vars1_prog2'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        c = [0.23]

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x

        self.sympy_eq = [
            c[0] * x[0]
        ]

@register_eq_class
class PopulationGrowthWithCarryingCapacity(KnownEquation):
    _eq_name = 'vars1_prog3'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        c = [0.79, 74.3]

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x

        self.sympy_eq = [
            c[0] * x[0] * (1 - x[0] / c[1])
        ]