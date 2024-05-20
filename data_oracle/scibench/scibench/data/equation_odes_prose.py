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
        self.b = 0.17

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)

    def np_eq(self, t, x):
            return np.array([
            np.sin(x[1]) - self.b * x[0],
            np.sin(x[2]) - self.b * x[1],
            np.sin(x[0]) - self.b * x[2],
        ])


@register_eq_class
class AizawaAttractor(KnownEquation):
    # Aizawa attractor
    _eq_name = 'Aizawa-attractor'
    _operator_set = ['add', 'sub', 'mul', 'n2','n3', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.a = 0.95
        self.b = 0.7
        self.c = 0.6
        self.d = 3.5
        self.e = 0.25
        self.f = 0.1

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)

    def np_eq(self, t, x):
            return np.array([
            (x[2] - self.b) * x[1] - self.d * x[1],
            self.d * x[0] - (x[2] - self.b) * x[1],
            self.c + self.a * x[2] - x[2] ** 3 / 3 - x[0] ** 2 + self.f * x[2] * x[0] ** 3,
        ])


@register_eq_class
class ChenLeeAttractor(KnownEquation):
    # Chen-Lee attractor
    _eq_name = 'Chen-Lee-attractor'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.a = 5
        self.d = -0.38

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)

    def np_eq(self, t, x):
            return np.array([
            self.a * x[0] - x[1] * x[2],
            -10 * x[1] + x[0] * x[2],
            self.d * x[2] + x[0] * x[1] / 3
        ])


@register_eq_class
class DadrasAttractor(KnownEquation):
    # Dadras attractor
    _eq_name = 'Dadras-attractor'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.a = 1.25
        self.b = 1.15
        self.c = 0.75
        self.d = 0.8
        self.e = 4

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)

    def np_eq(self, t, x):
        return np.array([
            x[1] / 2 - self.a * x[0] + self.b * x[1] * x[2],
            self.c * x[1] - x[0] * x[2] / 2 + x[2] / 2,
            self.d * x[0] * x[1] - self.e * x[2]
        ])


@register_eq_class
class RosslerAttractor(KnownEquation):
    # Dadras attractor
    _eq_name = 'Rossler-attractor'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.a = 0.1
        self.b = 0.1
        self.c = 14

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)

    def np_eq(self, t, x):
        return np.array([
            -x[1] - x[2],
            x[0] + self.a * x[1],
            self.b + x[2] * (x[1] - self.c)
        ])


@register_eq_class
class HalvorsenAttractor(KnownEquation):
    # Dadras attractor
    _eq_name = 'Halvorsen-attractor'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.a = -0.35

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)

    def np_eq(self, t, x):
        return np.array([
            self.a * x[0] - x[1] - x[2] - x[1] ** 2 / 4,
            self.a * x[1] - x[2] - x[0] - x[2] ** 2 / 4,
            self.a * x[2] - x[0] - x[1] - x[0] ** 2 / 4
        ])


@register_eq_class
class RabinovichFabrikantEquation(KnownEquation):
    # Dadras attractor
    _eq_name = 'Rabinovich–Fabrikant-equation'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.alpha = 0.98
        self.gamma = 0.1

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)

    def np_eq(self, t, x):
        return np.array([
            x[1] * (x[2] - 1 + x[0] ** 2) + self.gamma * x[0],
            x[0] * (3 * x[2] - 1 + x[1] ** 2) + self.gamma * x[1],
            -2 * x[2] * (self.alpha + x[0] * x[1])
        ])


@register_eq_class
class SprottLinzFAttractor(KnownEquation):
    # Dadras attractor
    _eq_name = 'Sprott-Linz-F-attractor'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.a = 0.5

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)

    def np_eq(self, t, x):
        return np.array([
            x[1] + x[2],
            -x[0] + self.a * x[1],
            x[1] ** 2 - x[2],
        ])


@register_eq_class
class FourWingChaoticAttractor(KnownEquation):
    # Dadras attractor
    _eq_name = 'Four-wing-chaotic-attractor'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.a = 0.2
        self.b = 0.01
        self.c = -0.4

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)

    def np_eq(self, t, x):
        return np.array([
            self.a * x[0] + x[1] * x[2],
            self.b * x[0] + self.c * x[1] - x[0] * x[2],
            -x[2] - x[0] * x[1],
        ])


@register_eq_class
class FourWingChaoticAttractor(KnownEquation):
    # Dadras attractor
    _eq_name = 'Four-wing-chaotic-attractor'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'cos']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.alpha = 0.2
        self.beta = 0.01
        self.gamma = -0.4
        self.delta = 0.02
        self.omega = 0.5

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)

    def np_eq(self, t, x):
        return np.array([
            1,
            x[2],
            -self.delta * x[2] - self.alpha * x[1] - self.beta * x[1] ** 3 + self.gamma * np.cos(self.omega * x[0]),
        ])
