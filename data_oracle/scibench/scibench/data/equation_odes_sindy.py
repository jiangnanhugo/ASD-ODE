import numpy as np

from scibench.data.base import KnownEquation, register_eq_class
from scibench.symbolic_data_generator import LogUniformSampling


@register_eq_class
class Lorenz(KnownEquation):
    _eq_name = 'Lorenz'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.sigma = 10
        self.beta = 8 / 3
        self.rho = 28

        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x=self.x
        self.sympy_eq=[
            self.sigma * (x[1] - x[0]),
            x[0] * (x[0] - self.rho - x[2]),
            x[0] * x[1] - self.beta * x[2]
        ]

    def np_eq(self, t, x):
        return np.array([
            self.sigma * (x[1] - x[0]),
            x[0] * (x[0] - self.rho - x[2]),
            x[0] * x[1] - self.beta * x[2]
        ])


@register_eq_class
class Glycolytic_oscillator(KnownEquation):
    _eq_name = 'Glycolytic_oscillator'
    _operator_set = ['add', 'sub', 'mul', 'div', 'n2', 'n3', 'n4', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.J0 = 2.5
        self.k1 = 100
        self.k2 = 6
        self.k3 = 16
        self.k4 = 100
        self.k5 = 1.28
        self.k6 = 12
        self.K = 1.8
        self.kappa = 13
        self.q = 4
        self.K1 = 0.52
        self.phi = 0.1
        self.N = 1
        self.A = 4

        vars_range_and_types = [LogUniformSampling((0.15, 1.6), only_positive=True),
                                LogUniformSampling((0.19, 2.16), only_positive=True),
                                LogUniformSampling((0.04, 0.20), only_positive=True),
                                LogUniformSampling((0.10, 0.35), only_positive=True),
                                LogUniformSampling((0.08, 0.30), only_positive=True),
                                LogUniformSampling((0.14, 2.67), only_positive=True),
                                LogUniformSampling((0.05, 0.10), only_positive=True)]
        super().__init__(num_vars=7, vars_range_and_types=vars_range_and_types)

    def np_eq(self, t, x):
        return np.array([
            self.J0 - (self.k1 * x[0] * x[5]) / (1 + (x[5] / self.K1) ** self.q),
            2 * (self.k1 * x[0] * x[5]) / (1 + (x[5] / self.K1) ** self.q) - self.k2 * x[1] * (N - x[4]) - self.k6 * x[
                1] * x[4],
            self.k2 * x[1] * (self.N - x[4]) - self.k3 * x[2] * (self.A - x[5]),
            self.k3 * x[2] * (self.A - x[5]) - self.k4 * x[3] * x[4] - self.kappa * (x[3] - x[6]),
            self.k2 * x[1] * (self.N - x[4]) - self.k4 * x[3] * x[4] - self.k6 * x[1] * x[4],
            -2 * self.k1 * x[0] * x[5] / (1 + (x[5] / self.K1) ** self.q) + 2 * self.k3 * x[2] * (
                        self.A - x[5]) - self.k5 * x[5],
            self.phi * self.kappa * (x[3] - x[6]) - self.K * x[6]
        ])


# Carbone and Veltri triadic MHD model
@register_eq_class
class mhd(KnownEquation):
    _eq_name = 'mhd'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.nu = 0.1
        self.mu = 0.2
        self.sigma = 0.3
        vars_range_and_types = [LogUniformSampling((0.001, 10), only_positive=True),
                                LogUniformSampling((0.001, 10), only_positive=True),
                                LogUniformSampling((0.001, 10), only_positive=True),
                                LogUniformSampling((0.001, 10), only_positive=True),
                                LogUniformSampling((0.001, 10), only_positive=True),
                                LogUniformSampling((0.001, 10), only_positive=True)]
        super().__init__(num_vars=6, vars_range_and_types=vars_range_and_types)

    def np_eq(self, t, x):
        return np.array([
            -2 * self.nu * x[0] + 4.0 * (x[1] * x[2] - x[4] * x[5]),
            -5 * self.nu * x[1] - 7.0 * (x[0] * x[2] - x[3] * x[5]),
            -9 * self.nu * x[2] + 3.0 * (x[0] * x[1] - x[3] * x[4]),
            -2 * self.mu * x[4] + 2.0 * (x[5] * x[1] - x[2] * x[4]),
            -5 * self.mu * x[4] + self.sigma * x[5] + 5.0 * (x[2] * x[3] - x[0] * x[5]),
            -9 * self.mu * x[5] + self.sigma * x[4] + 9.0 * (x[4] * x[0] - x[1] * x[3]),
        ])


@register_eq_class
class Pendulum_on_cart(KnownEquation):
    _eq_name = 'Pendulum_on_cart'
    _operator_set = ['add', 'sub', 'mul', 'div', 'n2', 'cos', 'sin', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.m = 1
        self.M = 1
        self.L = 1
        self.F = 0
        self.g = 9.81
        vars_range_and_types = [LogUniformSampling((0.001, 10), only_positive=True),
                                LogUniformSampling((0.001, 10), only_positive=True),
                                LogUniformSampling((0.001, 10), only_positive=True),
                                LogUniformSampling((0.001, 10), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=vars_range_and_types)

    def np_eq(self, t, x):
        return np.array(
            [x[2],
             x[3],
             (
                     (self.M + self.m) * g * np.sin(x[0])
                     - self.F * np.cos(x[0])
                     - self.m * self.L * np.sin(x[0]) * np.cos(x[0]) * x[2] ** 2
             ) / (
                     self.L * (self.M + self.m * np.sin(x[0]) ** 2)
             ),
             (self.m * self.L * np.sin(x[0]) * x[2] ** 2 + self.F - self.m * g * np.sin(x[0]) * np.cos(x[0])) / (
                     self.M + self.m * np.sin(x[0]) ** 2),
             ])


@register_eq_class
class Double_pendulum(KnownEquation):
    _eq_name = 'Double_pendulum'
    _operator_set = ['add', 'sub', 'mul', 'div', 'n2', 'sin', 'cos', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        m1 = 0.2704
        m2 = 0.2056
        a1 = 0.191
        a2 = 0.1621
        L1 = 0.2667
        L2 = 0.2667
        I1 = 0.003
        I2 = 0.0011
        g = 9.81
        k1 = 0
        k2 = 0

        self.vars_range_and_types = [LogUniformSampling((0.001, 10), only_positive=True),
                                     LogUniformSampling((0.001, 10), only_positive=True),
                                     LogUniformSampling((0.001, 10), only_positive=True),
                                     LogUniformSampling((0.001, 10), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x = self.x

        self.np_eq = [
            x[2],
            x[3],
            (
                    L1 * a2 ** 2 * g * m2 ** 2 * np.sin(x[0])
                    - 2 * L1 * a2 ** 3 * x[3] ** 2 * m2 ** 2 * np.sin(x[0] - x[1])
                    + 2 * I2 * L1 * g * m2 * np.sin(x[0])
                    + L1 * a2 ** 2 * g * m2 ** 2 * np.sin(x[0] - 2 * x[1])
                    + 2 * I2 * a1 * g * m1 * np.sin(x[0])
                    - (L1 * a2 * x[2] * m2) ** 2 * np.sin(2 * (x[0] - x[1]))
                    - 2 * I2 * L1 * a2 * x[3] ** 2 * m2 * np.sin(x[0] - x[1])
                    + 2 * a1 * a2 ** 2 * g * m1 * m2 * np.sin(x[0])
            ) / (
                    2 * I1 * I2
                    + (L1 * a2 * m2) ** 2
                    + 2 * I2 * L1 ** 2 * m2
                    + 2 * I2 * a1 ** 2 * m1
                    + 2 * I1 * a2 ** 2 * m2
                    - (L1 * a2 * m2) ** 2 * np.cos(2 * (x[0] - x[1]))
                    + 2 * (a1 * a2) ** 2 * m1 * m2
            ),
            (a2 * m2 * (
                    2 * I1 * g * np.sin(x[1])
                    + 2 * L1 ** 3 * x[2] ** 2 * m2 * np.sin(x[0] - x[1])
                    + 2 * L1 ** 2 * g * m2 * np.sin(x[1])
                    + 2 * I1 * L1 * x[2] ** 2 * np.sin(x[0] - x[1])
                    + 2 * a1 ** 2 * g * m1 * np.sin(x[1])
                    + L1 ** 2 * a2 * x[3] ** 2 * m2 * np.sin(2 * (x[0] - x[1]))
                    + 2 * L1 * a1 ** 2 * x[2] ** 2 * m1 * np.sin(x[0] - x[1])
                    - 2 * L1 ** 2 * g * m2 * np.cos(x[0] - x[1]) * np.sin(x[0])
                    - 2 * L1 * a1 * g * m1 * np.cos(x[0] - x[1]) * np.sin(x[0])
            )) / (
                    2 * (
                    I1 * I2
                    + (L1 * a2 * m2) ** 2
                    + I2 * L1 ** 2 * m2
                    + I2 * a1 ** 2 * m1
                    + I1 * a2 ** 2 * m2
                    - (L1 * a2 * m2) ** 2 * np.cos(x[0] - x[1]) ** 2
                    + a1 ** 2 * a2 ** 2 * m1 * m2
            )),
        ]

#
# # ---------------------
# # https://arxiv.org/pdf/2003.07140.pdf
# @register_eq_class
# class Glycolytic_a_simple_reaction_network(KnownEquation):
#     _eq_name = 'Glycolytic_a_simple_reaction_network'
#     _function_set = ['add', 'sub', 'mul', 'div', 'const']
#     expr_obj_thres = 1e-6
#
#     def __init__(self):
#         v1 = 2.5
#         k1 = 100
#         k_neg1 = 6
#
#         vars_range_and_types = [LogUniformSampling(0.15, 1.6, only_positive=True),
#                                 LogUniformSampling(0.19, 2.16, only_positive=True),
#                                 LogUniformSampling(0.04, 0.20, only_positive=True),
#                                 LogUniformSampling(0.10, 0.35, only_positive=True), ]
#         super().__init__(num_vars=7)
#         x = self.x
#         # v1 − k1s1x1 + k−1x2,
#         self.np_eqs = [
#             v1 - self.k1 * x[0] * x[2] + k_neg1 * x[3],
#             self.k2*x[1]-gamma*k3*s2**gamma*np.e+gmma *k_neg3*x[0] ]


# # %Lotka–Volterra equation
# # s
# @register_eq_class
# class Lotka_Volterra(KnownEquation):
#     _eq_name = 'Lotka–Volterra'
#     _function_set = ['add', 'sub', 'mul', 'div', 'const']
#     expr_obj_thres = 1e-6
#
#     def __init__(self):
#         # https://ulissigroup.cheme.cmu.edu/math-methods-chemical-engineering/notes/ordinary_differential_equations/19-linear-stability.html
#         alpha = 1
#         beta = 0.2
#         delta = 0.5
#         gamma = 0.2
#
#         vars_range_and_types = [LogUniformSampling(0.15, 1.6, only_positive=True),
#                                 LogUniformSampling(0.19, 2.16, only_positive=True),
#                                 LogUniformSampling(0.04, 0.20, only_positive=True),
#                                 LogUniformSampling(0.10, 0.35, only_positive=True), ]
#         super().__init__(num_vars=7)
#         x = self.x
#
#         self.np_eqs = [alpha * x[0] - beta * x[0] * x[1],
#                           delta * x[0] * x[1] - gamma * x[1]]


# # %Competitive Lotka–Volterra equation
# # https://en.wikipedia.org/wiki/Competitive_Lotka%E2%80%93Volterra_equations
#         # https://arxiv.org/pdf/2303.04919.pdf
# @register_eq_class
# class Competitive_Lotka_Volterra(KnownEquation):
#     _eq_name = 'Competitive-Lotka–Volterra'
#     _function_set = ['add', 'sub', 'mul', 'div', 'const']
#     expr_obj_thres = 1e-6
#
#     def __init__(self, N):
#         #  N is the total number of interacting species.
#         #  For simplicity all self-interacting terms αii are often set to 1.
#         alpha = np.random.rand(N, N)
#         r = np.random.rand(N)
#         np.fill_diagonal(alpha, 1.0)
#         #
#         # vars_range_and_types = [LogUniformSampling(0.15, 1.6, only_positive=True),
#         #                         LogUniformSampling(0.19, 2.16, only_positive=True),
#         #                         LogUniformSampling(0.04, 0.20, only_positive=True),
#         #                         LogUniformSampling(0.10, 0.35, only_positive=True), ]
#         super().__init__(num_vars=N)
#         x = self.x
#
#         self.np_eqs = []
#         for i in range(N):
#             summand = [alpha[i, j] * x[j] for j in range(N)]
#             self.np_eqs.append(r[i] * x[i] * (1 - sum(summand)))
