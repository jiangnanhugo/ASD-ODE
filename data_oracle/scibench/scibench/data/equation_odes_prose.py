import numpy as np

from scibench.data.base import KnownEquation, register_eq_class
from scibench.symbolic_data_generator import LogUniformSampling



# PROSE: Predicting Operators and Symbolic Expressions using
# Multimodal Transformers
# https://arxiv.org/pdf/2309.16816.pdf

@register_eq_class
class ThomasAttractor(KnownEquation):

    _eq_name = 'Thomas-attractor'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'sin']
    expr_obj_thres = 1e-6

    def __init__(self):
        b = 0.17

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x = self.x

        self.sympy_eq = [
            np.sin(x[1]) - b * x[0],
            np.sin(x[2]) - b * x[1],
            np.sin(x[0]) - b * x[2],
        ]


@register_eq_class
class AizawaAttractor(KnownEquation):
    # Aizawa attractor
    _eq_name = 'Aizawa-attractor'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        a = 0.95
        b = 0.7
        c = 0.6
        d = 3.5
        e = 0.25
        f = 0.1

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x = self.x

        self.sympy_eq = [
            (x[2] - b) * x[1] - d * x[1],
            d * x[0] - (x[2] - b) * x[1],
            c + a * x[2] - x[2] ** 3 / 3 - x[0] ** 2 + f * x[2] * x[0] ** 3,
        ]


@register_eq_class
class ChenLeeAttractor(KnownEquation):
    # Chen-Lee attractor
    _eq_name = 'Chen-Lee-attractor'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        a = 5
        d = -0.38

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x = self.x

        self.sympy_eq = [
            a * x[0] - x[1] * x[2],
            -10 * x[1] + x[0] * x[2],
            d * x[2] + x[0] * x[1] / 3
        ]


@register_eq_class
class DadrasAttractor(KnownEquation):
    # Dadras attractor
    _eq_name = 'Dadras-attractor'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        a = 1.25
        b = 1.15
        c = 0.75
        d = 0.8
        e = 4

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x = self.x

        self.sympy_eq = [
            x[1] / 2 - a * x[0] + b * x[1] * x[2],
            c * x[1] - x[0] * x[2] / 2 + x[2] / 2,
            d * x[0] * x[1] - e * x[2]
        ]


# TODO
# Rossler, Halvorsen, Rabinovich–Fabrikant, Sprott B, Sprott-Linz F, Four-wing chaotic, Duffing

@register_eq_class
class RosslerAttractor(KnownEquation):
    # Dadras attractor
    _eq_name = 'Rossler-attractor'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        a = 0.1
        b = 0.1
        c = 14

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x = self.x

        self.sympy_eq = [
            -x[1] - x[2],
            x[0] + a * x[1],
            b + x[2] * (x[1] - c)
        ]


@register_eq_class
class HalvorsenAttractor(KnownEquation):
    # Dadras attractor
    _eq_name = 'Halvorsen-attractor'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        a = -0.35

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x = self.x

        self.sympy_eq = [
            a * x[0] - x[1] - x[2] - x[1] ** 2 / 4,
            a * x[1] - x[2] - x[0] - x[2] ** 2 / 4,
            a * x[2] - x[0] - x[1] - x[0] ** 2 / 4
        ]


@register_eq_class
class RabinovichFabrikantEquation(KnownEquation):
    # Dadras attractor
    _eq_name = 'Rabinovich–Fabrikant-equation'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        alpha = 0.98
        gamma = 0.1

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x = self.x

        self.sympy_eq = [
            x[1] * (x[2] - 1 + x[0] ** 2) + gamma * x[0],
            x[0] * (3 * x[2] - 1 + x[1] ** 2) + gamma * x[1],
            -2 * x[2] * (alpha + x[0] * x[1])
        ]


@register_eq_class
class SprottLinzFAttractor(KnownEquation):
    # Dadras attractor
    _eq_name = 'Sprott-Linz-F-attractor'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        a = 0.5

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x = self.x

        self.sympy_eq = [
            x[1] + x[2],
            -x[0] + a * x[1],
            x[1] ** 2 - x[2],
        ]


@register_eq_class
class FourWingChaoticAttractor(KnownEquation):
    # Dadras attractor
    _eq_name = 'Four-wing-chaotic-attractor'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        a = 0.2
        b = 0.01
        c = -0.4

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x = self.x

        self.sympy_eq = [
            a * x[0] + x[1] * x[2],
            b * x[0] + c * x[1] - x[0] * x[2],
            -x[2] - x[0] * x[1],
        ]


@register_eq_class
class FourWingChaoticAttractor(KnownEquation):
    # Dadras attractor
    _eq_name = 'Four-wing-chaotic-attractor'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'cos']
    expr_obj_thres = 1e-6

    def __init__(self):
        alpha = 0.2
        beta = 0.01
        gamma = -0.4
        delta=0.02
        omega=0.5

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x = self.x

        self.sympy_eq = [
            1,
           x[2],
            -delta*x[2]-alpha*x[1]-beta*x[1]**3+gamma*np.cos(omega*x[0]),
        ]
