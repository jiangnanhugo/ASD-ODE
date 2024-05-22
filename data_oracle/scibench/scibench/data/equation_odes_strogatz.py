import numpy as np
from scibench.data.base import KnownEquation, register_eq_class
from scibench.symbolic_data_generator import LogUniformSampling, UniformSampling


@register_eq_class
class STROGATZ_P_20(KnownEquation):
    _eq_name = 'vars1_prog1'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "RC-circuit (charging capacitor)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['0.303030303030303 - 0.360750360750361*x[0]']

    def np_eq(self, t, x):
        return np.array([0.303030303030303 - 0.360750360750361 * x[0]])


@register_eq_class
class STROGATZ_P_22(KnownEquation):
    _eq_name = 'vars1_prog2'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Population growth (naive)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['0.23*x[0]']

    def np_eq(self, t, x):
        return np.array([0.23 * x[0]])


@register_eq_class
class STROGATZ_P_221(KnownEquation):
    _eq_name = 'vars1_prog3'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Population growth with carrying capacity"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[0]*(0.79 - 0.0106325706594886*x[0])']

    def np_eq(self, t, x):
        return np.array([x[0] * (0.79 - 0.0106325706594886 * x[0])])


@register_eq_class
class STROGATZ_P_38(KnownEquation):
    _eq_name = 'vars1_prog4'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'exp']
    _description = "RC-circuit with non-linear resistor (charging capacitor)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = [
            '(0.5*np.exp(1.04166666666667*x[0]) - 0.824360635350064)/(np.exp(1.04166666666667*x[0]) + 1.64872127070013)']

    def np_eq(self, t, x):
        return np.array([(0.5 * np.exp(x[0]) - 0.824360635350064) / (
                np.exp(x[0]) + 1.64872127070013)])


@register_eq_class
class STROGATZ_P_381(KnownEquation):
    _eq_name = 'vars1_prog5'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Velocity of a falling object with air resistance"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['9.81 - 0.0021175*x[0]**2']

    def np_eq(self, t, x):
        return np.array([9.81 - 0.0021175 * x[0] ** 2])


@register_eq_class
class STROGATZ_P_39(KnownEquation):
    _eq_name = 'vars1_prog6'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Autocatalysis with one fixed abundant chemical"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[0]*(2.1 - 0.5*x[0])']

    def np_eq(self, t, x):
        return np.array([x[0] * (2.1 - 0.5 * x[0])])


@register_eq_class
class STROGATZ_P_391(KnownEquation):
    _eq_name = 'vars1_prog7'
    _operator_set = ['add', 'sub', 'mul', 'const', 'log']
    _description = "Gompertz law for tumor growth"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[0]*(0.032*np.log(x[0]) + 0.0265136581621167)']

    def np_eq(self, t, x):
        return np.array([x[0] * (0.032 * np.log(x[0]) + 0.0265136581621167)])


@register_eq_class
class STROGATZ_P_392(KnownEquation):
    _eq_name = 'vars1_prog8'
    _operator_set = ['add', 'sub', 'mul', 'n2', 'const']
    _description = "Logistic equation with Allee effect"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[0]*(-0.000244755244755245*x[0]**2 + 0.0328951048951049*x[0] - 0.14)']

    def np_eq(self, t, x):
        return np.array([x[0] * (-0.000244755244755245 * x[0] ** 2 + 0.0328951048951049 * x[0] - 0.14)])


@register_eq_class
class STROGATZ_P_40(KnownEquation):
    _eq_name = 'vars1_prog9'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Language death model for two languages"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['0.32 - 0.6*x[0]']

    def np_eq(self, t, x):
        return np.array([0.32 - 0.6 * x[0]])


@register_eq_class
class STROGATZ_P_401(KnownEquation):
    _eq_name = 'vars1_prog10'
    _operator_set = ['add', 'sub', 'mul', 'div', 'n2', 'const']
    _description = "Refined language death model for two languages"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 0.95), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['-0.8*x[0]*(1 - x[0])**2 + 0.2*x[0]**2 - 0.2*x[0]**2']

    def np_eq(self, t, x):
        return np.array([-0.8 * x[0] * (1 - x[0]) ** 1 + 0.2 * x[0] ** 2 - 0.2 * x[0] ** 2])


@register_eq_class
class STROGATZ_P_41(KnownEquation):
    _eq_name = 'vars1_prog11'
    _operator_set = ['add', 'sub', 'mul', 'n3', 'const']
    _description = "Naive critical slowing down (statistical mechanics)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['-x[0]**3']

    def np_eq(self, t, x):
        return np.array([-x[0] ** 3])


@register_eq_class
class STROGATZ_P_55(KnownEquation):
    _eq_name = 'vars1_prog12'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Photons in a laser (simple)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[0]*(1.8 - 0.1107*x[0])']

    def np_eq(self, t, x):
        return np.array([x[0] * (1.8 - 0.1107 * x[0])])


@register_eq_class
class STROGATZ_P_63(KnownEquation):
    _eq_name = 'vars1_prog13'
    _operator_set = ['add', 'sub', 'mul', 'const', 'sin', 'cos']
    _description = "Overdamped bead on a rotating hoop"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['(0.95157*np.cos(x[0]) - 0.0981)*np.sin(x[0])']

    def np_eq(self, t, x):
        return np.array([(0.95157 * np.cos(x[0]) - 0.0981) * np.sin(x[0])])


@register_eq_class
class STROGATZ_P_75(KnownEquation):
    _eq_name = 'vars1_prog14'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Budworm outbreak model with predation"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[0]*(-0.9*x[0] + (0.78 - 0.00962962962962963*x[0])*(x[0]**2 + 449.44))/(x[0]**2 + 449.44)']

    def np_eq(self, t, x):
        return np.array(
            [x[0] * (-0.9 * x[0] + (0.78 - 0.00962962962962963 * x[0]) * (x[0] ** 2 + 449.44)) / (x[0] ** 2 + 449.44)])


@register_eq_class
class STROGATZ_P_76(KnownEquation):
    _eq_name = 'vars1_prog15'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Budworm outbreak with predation (dimensionless)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[0]*(-x[0] + (0.4 - 0.00421052631578947*x[0])*(x[0]**2 + 1))/(x[0]**2 + 1)']

    def np_eq(self, t, x):
        return np.array([x[0] * (-x[0] + (0.4 - 0.00421052631578947 * x[0]) * (x[0] ** 2 + 1)) / (x[0] ** 2 + 1)])


@register_eq_class
class STROGATZ_P_87(KnownEquation):
    _eq_name = 'vars1_prog16'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const','n2']
    _description = "Landau equation (typical time scale tau = 1)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[0]*(-0.001*x[0]**4 + 0.04*x[0]**2 + 0.1)']

    def np_eq(self, t, x):
        return np.array([x[0] * (-0.001 * x[0] ** 4 + 0.04 * x[0] ** 2 + 0.1)])


@register_eq_class
class STROGATZ_P_89(KnownEquation):
    _eq_name = 'vars1_prog17'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Logistic equation with harvesting/fishing"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['-0.004*x[0]**2 + 0.4*x[0] - 0.3']

    def np_eq(self, t, x):
        return np.array([-0.004 * x[0] ** 2 + 0.4 * x[0] - 0.3])


@register_eq_class
class STROGATZ_P_90(KnownEquation):
    _eq_name = 'vars1_prog18'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Improved logistic equation with harvesting/fishing"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[0]*((0.4 - 0.004*x[0])*(x[0] + 50.0) - 0.24)/(x[0] + 50.0)']

    def np_eq(self, t, x):
        return np.array([x[0] * ((0.4 - 0.004 * x[0]) * (x[0] + 50.0) - 0.24) / (x[0] + 50.0)])


@register_eq_class
class STROGATZ_P_901(KnownEquation):
    _eq_name = 'vars1_prog19'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Improved logistic equation with harvesting/fishing (dimensionless)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[0]*((1 - x[0])*(x[0] + 0.8) - 0.08)/(x[0] + 0.8)']

    def np_eq(self, t, x):
        return np.array([x[0] * ((1 - x[0]) * (x[0] + 0.8) - 0.08) / (x[0] + 0.8)])


@register_eq_class
class STROGATZ_P_91(KnownEquation):
    _eq_name = 'vars1_prog20'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'n2']
    _description = "Autocatalytic gene switching (dimensionless)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['(x[0]**2 + (0.1 - 0.55*x[0])*(x[0]**2 + 1))/(x[0]**2 + 1)']

    def np_eq(self, t, x):
        return np.array([(x[0] ** 2 + (0.1 - 0.55 * x[0]) * (x[0] ** 2 + 1)) / (x[0] ** 2 + 1)])


@register_eq_class
class STROGATZ_P_92(KnownEquation):
    _eq_name = 'vars1_prog21'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'negexp']
    _description = "Dimensionally reduced SIR infection model for dead people (dimensionless)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['-0.2*x[0] + 1.2 - np.exp(-x[0])']

    def np_eq(self, t, x):
        return np.array([-0.2 * x[0] + 1.2 - np.exp(-x[0])])


@register_eq_class
class STROGATZ_P_93(KnownEquation):
    _eq_name = 'vars1_prog22'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Hysteretic activation of a protein expression (positive feedback, basal promoter expression)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['(0.4*x[0]**5 + (1.4 - 0.89*x[0])*(x[0]**5 + 123.0))/(x[0]**5 + 123.0)']

    def np_eq(self, t, x):
        return np.array([(0.4 * x[0] ** 5 + (1.4 - 0.89 * x[0]) * (x[0] ** 5 + 123.0)) / (x[0] ** 5 + 123.0)])


@register_eq_class
class STROGATZ_P_104(KnownEquation):
    _eq_name = 'vars1_prog23'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'sin']
    _description = "Overdamped pendulum with constant driving torque/fireflies/Josephson junction (dimensionless)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['0.21 - np.sin(x[0])']

    def np_eq(self, t, x):
        return np.array([0.21 - np.sin(x[0])])


@register_eq_class
class STROGATZ_P_126(KnownEquation):
    _eq_name = 'vars2_prog1'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Harmonic oscillator without damping"

    def __init__(self):
        self.vars_range_and_types = [UniformSampling((-10, 10.0), only_positive=True),
                                     UniformSampling((-10, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[1]', '-2.1*x[0]']

    def np_eq(self, t, x):
        return np.array([x[1], -2.1 * x[0]])


@register_eq_class
class STROGATZ_P_144(KnownEquation):
    _eq_name = 'vars2_prog2'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Harmonic oscillator with damping"

    def __init__(self):
        self.vars_range_and_types = [UniformSampling((-10, 10.0), only_positive=True),
                                     UniformSampling((-10, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[1]', '-4.5*x[0] - 0.43*x[1]']

    def np_eq(self, t, x):
        return np.array([x[1], -4.5 * x[0] - 0.43 * x[1]])


@register_eq_class
class STROGATZ_P_157(KnownEquation):
    _eq_name = 'vars2_prog3'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Lotka-Volterra competition model (Strogatz version)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 4.0), only_positive=True),
                                     LogUniformSampling((1e-2, 3.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[0]*(-x[0] - 2.0*x[1] + 3.0)', 'x[1]*(-x[0] - x[1] + 2.0)']

    def np_eq(self, t, x):
        return np.array([x[0] * (-x[0] - 2.0 * x[1] + 3.0), x[1] * (-x[0] - x[1] + 2.0)])


@register_eq_class
class LOTKA_VOLTERRA(KnownEquation):
    _eq_name = 'vars2_prog4'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Lotka-Volterra simple (as on Wikipedia)"

    def __init__(self):
        self.vars_range_and_types = [UniformSampling((-10, 10.0), only_positive=True),
                                     UniformSampling((-10, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[0]*(1.84 - 1.45*x[1])', 'x[1]*(1.62*x[0] - 3.0)']

    def np_eq(self, t, x):
        return np.array([x[0] * (1.84 - 1.45 * x[1]), x[1] * (1.62 * x[0] - 3.0)])


@register_eq_class
class STROGATZ_P_169(KnownEquation):
    _eq_name = 'vars2_prog5'
    _operator_set = ['add', 'sub', 'mul', 'const', 'sin']
    _description = "Pendulum without friction"

    def __init__(self):
        self.vars_range_and_types = [UniformSampling((-10, 10.0), only_positive=True),
                                     UniformSampling((-10, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[1]', '-0.9*np.sin(x[0])']

    def np_eq(self, t, x):
        return np.array([x[1], -0.9 * np.sin(x[0])])


@register_eq_class
class STROGATZ_P_181(KnownEquation):
    _eq_name = 'vars2_prog6'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Dipole fixed point"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((0.001, 10.0), only_positive=True),
                                     UniformSampling((-3, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['0.65*x[0]*x[1]', '-x[0]**2 + x[1]**2']

    def np_eq(self, t, x):
        return np.array([0.65 * x[0] * x[1], -x[0] ** 2 + x[1] ** 2])


@register_eq_class
class STROGATZ_P_187(KnownEquation):
    _eq_name = 'vars2_prog7'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "RNA molecules catalyzing each others replication"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 0.6), only_positive=True),
                                     UniformSampling((1e-2, 0.6), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[0]*x[1]*(1 - 1.61*x[0])', 'x[0]*x[1]*(1 - 1.61*x[1])']

    def np_eq(self, t, x):
        return np.array([x[0] * x[1] * (1 - 1.61 * x[0]), x[0] * x[1] * (1 - 1.61 * x[1])])


@register_eq_class
class STROGATZ_P_188(KnownEquation):
    _eq_name = 'vars2_prog8'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "SIR infection model only for healthy and sick"

    def __init__(self):
        self.vars_range_and_types = [UniformSampling((-10, 10.0), only_positive=True),
                                     UniformSampling((-10, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['-0.4*x[0]*x[1]', 'x[1]*(0.4*x[0] - 0.314)']

    def np_eq(self, t, x):
        return np.array([-0.4 * x[0] * x[1], x[1] * (0.4 * x[0] - 0.314)])


@register_eq_class
class STROGATZ_P_190(KnownEquation):
    _eq_name = 'vars2_prog9'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Damped double well oscillator"

    def __init__(self):
        self.vars_range_and_types = [UniformSampling((-5, 5.0), only_positive=True),
                                     UniformSampling((-10, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[1]', '-x[0]**3 + x[0] - 0.18*x[1]']

    def np_eq(self, t, x):
        return np.array([x[1], -x[0] ** 3 + x[0] - 0.18 * x[1]])


@register_eq_class
class STROGATZ_P_1901(KnownEquation):
    _eq_name = 'vars2_prog10'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'sin', 'cos']
    _description = "Glider (dimensionless)"

    def __init__(self):
        self.vars_range_and_types = [UniformSampling((-10, 10.0), only_positive=True),
                                     UniformSampling((-10, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['-0.08*x[0]**2 - sin(x[1])', 'x[0] - cos(x[1])/x[0]']

    def np_eq(self, t, x):
        return np.array([-0.08 * x[0] ** 2 - np.sin(x[1]), x[0] - np.cos(x[1]) / x[0]])


@register_eq_class
class STROGATZ_P_191(KnownEquation):
    _eq_name = 'vars2_prog11'
    _operator_set = ['add', 'sub', 'mul', 'const', 'sin', 'cos']
    _description = "Frictionless bead on a rotating hoop"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((-10, 10.0), only_positive=True),
                                     LogUniformSampling((-4, 4.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[1]', '(cos(x[0]) - 0.93)*sin(x[0])']

    def np_eq(self, t, x):
        return np.array([x[1], (np.cos(x[0]) - 0.93) * np.sin(x[0])])


@register_eq_class
class STROGATZ_P_194(KnownEquation):
    _eq_name = 'vars2_prog12'
    _operator_set = ['add', 'sub', 'mul', 'n2', 'const', 'sin', 'cos', 'cot']
    _description = "Rotational dynamics of an object in a shear flow"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['np.cos(x[0])*cot(x[1])', '(3.2*np.sin(x[1])**2 + 1.0)*np.sin(x[0])']

    def np_eq(self, t, x):
        return np.array([np.cos(x[0]) / np.tan(x[1]), (3.2 * np.sin(x[1]) ** 2 + 1.0) * np.sin(x[0])])


@register_eq_class
class STROGATZ_P_195(KnownEquation):
    _eq_name = 'vars2_prog13'
    _operator_set = ['add', 'sub', 'mul', 'const', 'sin', 'cos']
    _description = "Pendulum with non-linear damping, no driving"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[1]', '-0.07*x[1]*np.cos(x[0]) - x[1] - np.sin(x[0])']

    def np_eq(self, t, x):
        return np.array([x[1], -0.07 * x[1] * np.cos(x[0]) - x[1] - np.sin(x[0])])


@register_eq_class
class STROGATZ_P_200(KnownEquation):
    _eq_name = 'vars2_prog14'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Van der Pol oscillator (standard form)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[1]', '-0.43*x[0]**2*x[1] - x[0] + 0.43*x[1]']

    def np_eq(self, t, x):
        return np.array([x[1], -0.43 * x[0] ** 2 * x[1] - x[0] + 0.43 * x[1]])


@register_eq_class
class STROGATZ_P_214(KnownEquation):
    _eq_name = 'vars2_prog15'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Van der Pol oscillator (simplified form)"

    def __init__(self):
        self.vars_range_and_types = [UniformSampling((-10, 10.0), only_positive=True),
                                     UniformSampling((-10, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['-1.12333333333333*x[0]**3 + 3.37*x[0] + 3.37*x[1]', '-0.29673590504451*x[0]']

    def np_eq(self, t, x):
        return np.array([-1.12333333333333 * x[0] ** 3 + 3.37 * x[0] + 3.37 * x[1], -0.29673590504451 * x[0]])


@register_eq_class
class STROGATZ_P_207(KnownEquation):
    _eq_name = 'vars2_prog16'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Glycolytic oscillator, e.g., ADP and F6P in yeast"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[0]**2*x[1] - x[0] + 2.4*x[1]', '-x[0]**2*x[1] - 2.4*x[0] + 0.07']

    def np_eq(self, t, x):
        return np.array([x[0] ** 2 * x[1] - x[0] + 2.4 * x[1], -x[0] ** 2 * x[1] - 2.4 * x[0] + 0.07])


@register_eq_class
class STROGATZ_P_217(KnownEquation):
    _eq_name = 'vars2_prog17'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Duffing equation (weakly non-linear oscillation)"

    def __init__(self):
        self.vars_range_and_types = [UniformSampling((-10, 10.0), only_positive=True),
                                     UniformSampling((-10, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[1]', '-0.886*x[0]**2*x[1] - x[0] + 0.886*x[1]']

    def np_eq(self, t, x):
        return np.array([x[1], -0.886 * x[0] ** 2 * x[1] - x[0] + 0.886 * x[1]])


@register_eq_class
class STROGATZ_P_238(KnownEquation):
    _eq_name = 'vars2_prog18'
    _operator_set = ['add', 'sub', 'mul', 'n2', 'n3', 'const']
    _description = "Cell cycle model by Tyson for interaction between protein cdc2 and cyclin"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['-15.3*x[0]**3 + 15.3*x[0]**2*x[1] - 1.0153*x[0] + 0.0153*x[1]', '0.3 - x[0]']

    def np_eq(self, t, x):
        return np.array([-15.3 * x[0] ** 3 + 15.3 * x[0] ** 2 * x[1] - 1.0153 * x[0] + 0.0153 * x[1], 0.3 - x[0]])


@register_eq_class
class STROGATZ_P_260(KnownEquation):
    _eq_name = 'vars2_prog19'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const','n2']
    _description = "Reduced model for chlorine dioxide-iodine-malonic acid reaction"

    def __init__(self):
        self.vars_range_and_types = [UniformSampling((-8, 8), only_positive=True),
                                     UniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['(-4.0*x[0]*x[1] + (8.9 - x[0])*(x[0]**2 + 1))/(x[0]**2 + 1)',
                         '1.4*x[0]*(x[0]**2 - x[1] + 1)/(x[0]**2 + 1)']

    def np_eq(self, t, x):
        return np.array([(-4.0 * x[0] * x[1] + (8.9 - x[0]) * (x[0] ** 2 + 1)) / (x[0] ** 2 + 1),
                         1.4 * x[0] * (x[0] ** 2 - x[1] + 1) / (x[0] ** 2 + 1)])


@register_eq_class
class STROGATZ_P_269(KnownEquation):
    _eq_name = 'vars2_prog20'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'sin']
    _description = "Driven pendulum with linear damping / Josephson junction"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[1]', '-0.64*x[1] - np.sin(x[0]) + 1.67']

    def np_eq(self, t, x):
        return np.array([x[1], -0.64 * x[1] - np.sin(x[0]) + 1.67])


@register_eq_class
class STROGATZ_P_300(KnownEquation):
    _eq_name = 'vars2_prog21'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'sin']
    _description = "Driven pendulum with quadratic damping (dimensionless)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[1]', '-0.64*x[1]*np.abs(x[1]) - np.sin(x[0]) + 1.67']

    def np_eq(self, t, x):
        return np.array([x[1],
                         -0.64 * x[1] * x[1] - np.sin(x[0]) + 1.67])


@register_eq_class
class STROGATZ_P_288(KnownEquation):
    _eq_name = 'vars2_prog22'
    _operator_set = ['add', 'sub', 'mul', 'n2', 'const']
    _description = "Isothermal autocatalytic reaction model"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['-x[0]*x[1]**2 - 0.5*x[0] + 0.5', 'x[1]*(x[0]*x[1] - 0.02)']

    def np_eq(self, t, x):
        return np.array([-x[0] * x[1] ** 2 - 0.5 * x[0] + 0.5, x[1] * (x[0] * x[1] - 0.02)])


@register_eq_class
class STROGATZ_P_289(KnownEquation):
    _eq_name = 'vars2_prog23'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'sin']
    _description = "Interacting bar magnets"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['-np.sin(x[0]) + 0.33*np.sin(x[0] - x[1])', '-np.sin(x[1]) - 0.33*np.sin(x[0] - x[1])']

    def np_eq(self, t, x):
        return np.array([-np.sin(x[0]) + 0.33 * np.sin(x[0] - x[1]), -np.sin(x[1]) - 0.33 * np.sin(x[0] - x[1])])


@register_eq_class
class STROGATZ_P_293(KnownEquation):
    _eq_name = 'vars2_prog24'
    _operator_set = ['add', 'sub', 'mul', 'n2', 'const']
    _description = "Bacterial respiration model for nutrients and oxygen levels"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['(-x[0]*x[1] + (18.3 - x[0])*(0.48*x[0]**2 + 1))/(0.48*x[0]**2 + 1)',
                         '(5.3904*x[0]**2 - x[0]*x[1] + 11.23)/(0.48*x[0]**2 + 1)']

    def np_eq(self, t, x):
        return np.array([(-x[0] * x[1] + (18.3 - x[0]) * (0.48 * x[0] ** 2 + 1)) / (0.48 * x[0] ** 2 + 1),
                         (5.3904 * x[0] ** 2 - x[0] * x[1] + 11.23) / (0.48 * x[0] ** 2 + 1)])


@register_eq_class
class STROGATZ_P_296(KnownEquation):
    _eq_name = 'vars2_prog25'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Brusselator: hypothetical chemical oscillation model (dimensionless)"

    def __init__(self):
        self.vars_range_and_types = [UniformSampling((-10, 10.0), only_positive=True),
                                     UniformSampling((-10, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['3.1*x[0]**2*x[1] - 4.03*x[0] + 1', 'x[0]*(-3.1*x[0]*x[1] + 3.03)']

    def np_eq(self, t, x):
        return np.array([3.1 * x[0] ** 2 * x[1] - 4.03 * x[0] + 1, x[0] * (-3.1 * x[0] * x[1] + 3.03)])


@register_eq_class
class STROGATZ_P_2961(KnownEquation):
    _eq_name = 'vars2_prog26'
    _operator_set = ['add', 'sub', 'mul','n2', 'const']
    _description = "Chemical oscillator model by Schnackenberg 1979 (dimensionless)"

    def __init__(self):
        self.vars_range_and_types = [UniformSampling((-5, 5.0), only_positive=True),
                                     UniformSampling((-5, 5.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[0]**2*x[1] - x[0] + 0.24', '-x[0]**2*x[1] + 1.43']

    def np_eq(self, t, x):
        return np.array([x[0] ** 2 * x[1] - x[0] + 0.24, -x[0] ** 2 * x[1] + 1.43])


@register_eq_class
class STROGATZ_P_301(KnownEquation):
    _eq_name = 'vars2_prog27'
    _operator_set = ['add', 'sub', 'mul', 'const', 'sin', 'cos']
    _description = "Oscillator death model by Ermentrout and Kopell 1990"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['np.sin(x[1])*np.cos(x[0]) + 1.432', 'np.sin(x[1])*np.cos(x[0]) + 0.972']

    def np_eq(self, t, x):
        return np.array([np.sin(x[1]) * np.cos(x[0]) + 1.432, np.sin(x[1]) * np.cos(x[0]) + 0.972])


@register_eq_class
class STROGATZ_P_82(KnownEquation):
    _eq_name = 'vars3_prog1'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Maxwell-Bloch equations (laser dynamics)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['-0.1*x[0] + 0.1*x[1]', '0.21*x[0]*x[2] - 0.21*x[1]', '-1.054*x[0]*x[1] - 0.34*x[2] + 1.394']

    def np_eq(self, t, x):
        return np.array(
            [-0.1 * x[0] + 0.1 * x[1], 0.21 * x[0] * x[2] - 0.21 * x[1], -1.054 * x[0] * x[1] - 0.34 * x[2] + 1.394])


@register_eq_class
class MODEL_FOR_APOPTOSIS(KnownEquation):
    _eq_name = 'vars3_prog2'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Model for apoptosis (cell death)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['(-0.4*x[0]*x[1] + (0.1 - 0.05*x[0])*(x[0] + 0.1))/(x[0] + 0.1)',
                         '(-7.95*x[0]*x[1]*(x[1] + 0.1) - 0.2*x[1]*(x[1] + 2.0) + x[2]*(0.6*x[1] + 0.06)*(x[1] + 0.1)*(x[1] + 2.0))/((x[1] + 0.1)*(x[1] + 2.0))',
                         '(7.95*x[0]*x[1]*(x[1] + 0.1) + 0.2*x[1]*(x[1] + 2.0) - x[2]*(0.6*x[1] + 0.06)*(x[1] + 0.1)*(x[1] + 2.0))/((x[1] + 0.1)*(x[1] + 2.0))']

    def np_eq(self, t, x):
        return np.array([(-0.4 * x[0] * x[1] + (0.1 - 0.05 * x[0]) * (x[0] + 0.1)) / (x[0] + 0.1),
                         (-7.95 * x[0] * x[1] * (x[1] + 0.1) - 0.2 * x[1] * (x[1] + 2.0) + x[2] * (0.6 * x[1] + 0.06) * (
                x[1] + 0.1) * (x[1] + 2.0)) / ((x[1] + 0.1) * (x[1] + 2.0)),
                         (7.95 * x[0] * x[1] * (x[1] + 0.1) + 0.2 * x[1] * (x[1] + 2.0) - x[2] * (
                                 0.6 * x[1] + 0.06) * (x[1] + 0.1) * (x[1] + 2.0)) / ((x[1] + 0.1) * (x[1] + 2.0))])


@register_eq_class
class STROGATZ_P_319(KnownEquation):
    _eq_name = 'vars3_prog3'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Lorenz equations in well-behaved periodic regime"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['-5.1*x[0] + 5.1*x[1]', '-x[0]*x[2] + 12.0*x[0] - x[1]', 'x[0]*x[1] - 1.67*x[2]']

    def np_eq(self, t, x):
        return np.array([-5.1 * x[0] + 5.1 * x[1], -x[0] * x[2] + 12.0 * x[0] - x[1], x[0] * x[1] - 1.67 * x[2]])


@register_eq_class
class STROGATZ_P_3194(KnownEquation):
    _eq_name = 'vars3_prog4'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Lorenz equations in complex periodic regime"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['-10.0*x[0] + 10.0*x[1]', '-x[0]*x[2] + 99.96*x[0] - x[1]',
                         'x[0]*x[1] - 2.6666666666666665*x[2]']

    def np_eq(self, t, x):
        return np.array(
            [-10.0 * x[0] + 10.0 * x[1],
             -x[0] * x[2] + 99.96 * x[0] - x[1], x[0] * x[1] - 2.67 * x[2]])


@register_eq_class
class STROGATZ_P_3195(KnownEquation):
    _eq_name = 'vars3_prog5'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Lorenz equations standard parameters (chaotic)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['-10.0*x[0] + 10.0*x[1]', '-x[0]*x[2] + 28.0*x[0] - x[1]',
                         'x[0]*x[1] - 2.6666666666666665*x[2]']

    def np_eq(self, t, x):
        return np.array(
            [-10.0 * x[0] + 10.0 * x[1],
             -x[0] * x[2] + 28.0 * x[0] - x[1], x[0] * x[1] - 2.67 * x[2]])


@register_eq_class
class ROSSLER_ATTRACTOR(KnownEquation):
    _eq_name = 'vars3_prog6'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Rössler attractor (stable fixed point)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['-5.0*x[1] - 5.0*x[2]', '5.0*x[0] - 1.0*x[1]', '5.0*x[0]*x[2] - 28.5*x[2] + 1.0']

    def np_eq(self, t, x):
        return np.array([-5.0 * x[1] - 5.0 * x[2], 5.0 * x[0] - 1.0 * x[1], 5.0 * x[0] * x[2] - 28.5 * x[2] + 1.0])


@register_eq_class
class ROSSLER_ATTRACTOR_PERIODIC(KnownEquation):
    _eq_name = 'vars3_prog7'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Rossler attractor (periodic)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['-5.0*x[1] - 5.0*x[2]', '5.0*x[0] + 0.5*x[1]', '5.0*x[0]*x[2] - 28.5*x[2] + 1.0']

    def np_eq(self, t, x):
        return np.array([-5.0 * x[1] - 5.0 * x[2], 5.0 * x[0] + 0.5 * x[1], 5.0 * x[0] * x[2] - 28.5 * x[2] + 1.0])


@register_eq_class
class ROSSLER_ATTRACTOR_CHAOTIC(KnownEquation):
    _eq_name = 'vars3_prog8'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Rössler attractor (chaotic)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['-5.0*x[1] - 5.0*x[2]', '5.0*x[0] + 1.0*x[1]', '5.0*x[0]*x[2] - 28.5*x[2] + 1.0']

    def np_eq(self, t, x):
        return np.array([-5.0 * x[1] - 5.0 * x[2],
                         5.0 * x[0] + 1.0 * x[1], 5.0 * x[0] * x[2] - 28.5 * x[2] + 1.0])


@register_eq_class
class AIZAWA_ATTRACTOR_CHAOTIC(KnownEquation):
    _eq_name = 'vars3_prog9'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Aizawa attractor (chaotic)"

    def __init__(self):
        self.vars_range_and_types = [UniformSampling((-1.5, 1.5), only_positive=True),
                                     UniformSampling((-1.5, 1.5), only_positive=True),
                                     UniformSampling((-0.5, 1.5), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['x[0]*x[2] - 0.7*x[0] - 3.5*x[1]', '3.5*x[0] + x[1]*x[2] - 0.7*x[1]',
                         '0.1*x[0]**3*x[2] - 0.25*x[0]**2*x[2] - x[0]**2 - 0.25*x[1]**2*x[2] - x[1]**2 - 0.33*x[2]**3 + 0.95*x[2] + 0.65']

    def np_eq(self, t, x):
        return np.array([x[0] * x[2] - 0.7 * x[0] - 3.5 * x[1],
                         3.5 * x[0] + x[1] * x[2] - 0.7 * x[1],
                         0.1 * x[0] ** 3 * x[2] - 0.25 * x[0] ** 2 * x[2] - x[0] ** 2 - 0.25 * x[1] ** 2 * x[2] - x[1] ** 2 - 0.33 * x[2] ** 3 + 0.95 * x[2] + 0.65])

# PROSE: Predicting Operators and Symbolic Expressions using Multimodal Transformers
# https://arxiv.org/pdf/2309.16816.pdf
@register_eq_class
class AizawaAttractor(KnownEquation):
    # Aizawa attractor
    _eq_name = 'vars3_prog10'
    _operator_set = ['add', 'sub', 'mul', 'n2', 'n3', 'const']
    expr_obj_thres = 1e-6
    _description = "Aizawa attractor"
    def __init__(self):
        self.a = 0.95
        self.b = 0.7
        self.c = 0.6
        self.d = 3.5
        self.e = 0.25
        self.f = 0.1

        self.vars_range_and_types = [UniformSampling((-1.5, 1.5), only_positive=True),
                                     UniformSampling((-1.5, 1.5), only_positive=True),
                                     UniformSampling((-0.5, 1.5), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)

    def np_eq(self, t, x):
        return np.array([
            (x[2] - self.b) * x[1] - self.d * x[1],
            self.d * x[0] - (x[2] - self.b) * x[1],
            self.c + self.a * x[2] - x[2] ** 3 / 3 - x[0] ** 2 + self.f * x[2] * x[0] ** 3,
        ])


@register_eq_class
class CHEN_LEE_ATTRACTOR(KnownEquation):
    _eq_name = 'vars3_prog11'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Chen-Lee attractor"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['5.0*x[0] - x[1]*x[2]', 'x[0]*x[2] - 10.0*x[1]', '0.333333333333333*x[0]*x[1] - 3.8*x[2]']

    def np_eq(self, t, x):
        return np.array(
            [5.0 * x[0] - x[1] * x[2],
             x[0] * x[2] - 10.0 * x[1],
             0.333333333333333 * x[0] * x[1] - 3.8 * x[2]])


@register_eq_class
class DadrasAttractor(KnownEquation):
    # Dadras attractor
    _eq_name = 'vars3_prog12'
    _operator_set = ['add', 'sub', 'mul', 'const']
    expr_obj_thres = 1e-6
    _description = "attractor"

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
    _eq_name = 'vars3_prog13'
    _operator_set = ['add', 'sub', 'mul',  'const']
    expr_obj_thres = 1e-6
    _description = "attractor"

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
    _eq_name = 'vars3_prog14'
    _operator_set = ['add', 'sub', 'mul', 'n2', 'const']
    expr_obj_thres = 1e-6
    _description = "attractor"

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
    _eq_name = 'vars3_prog15'
    _operator_set = ['add', 'sub', 'mul', 'n2', 'const']
    expr_obj_thres = 1e-6
    _description = "attractor"

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
    _eq_name = 'vars3_prog16'
    _operator_set = ['add', 'sub', 'mul', 'n2', 'const']
    expr_obj_thres = 1e-6
    _description = "attractor"

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
    _eq_name = 'vars3_prog17'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6
    _description = "attractor"

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
    _eq_name = 'vars3_prog18'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'cos']
    expr_obj_thres = 1e-6
    _description = "attractor"

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


@register_eq_class
class ThomasAttractor(KnownEquation):
    _eq_name = 'vars3_prog19'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'sin']
    expr_obj_thres = 1e-6
    _description = "attractor"
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
class STROGATZ_P_295(KnownEquation):
    _eq_name = 'vars4_prog1'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'exp']
    _description = "Binocular rivalry model with adaptation (oscillations)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = [
            '(-x[0]*(0.246596963941606*np.exp(0.4*x[1] + 0.89*x[2]) + 1) + 1)/(0.246596963941606*np.exp(0.4*x[1] + 0.89*x[2]) + 1)',
            '1.0*x[0] - 1.0*x[1]',
            '(-x[2]*(0.246596963941606*np.exp(0.89*x[0] + 0.4*x[3]) + 1) + 1)/(0.246596963941606*np.exp(0.89*x[0] + 0.4*x[3]) + 1)',
            '1.0*x[2] - 1.0*x[3]']

    def np_eq(self, t, x):
        return np.array([(-x[0] * (0.246596963941606 * np.exp(0.4 * x[1] + 0.89 * x[2]) + 1) + 1) / (
                0.246596963941606 * np.exp(0.4 * x[1] + 0.89 * x[2]) + 1), 1.0 * x[0] - 1.0 * x[1],
                         (-x[2] * (0.246596963941606 * np.exp(0.89 * x[0] + 0.4 * x[3]) + 1) + 1) / (
                                 0.246596963941606 * np.exp(0.89 * x[0] + 0.4 * x[3]) + 1),
                         1.0 * x[2] - 1.0 * x[3]])


@register_eq_class
class SEIR_INFECTION_MODEL_PROPORTIONS(KnownEquation):
    _eq_name = 'vars4_prog2'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "SEIR infection model (proportions)"

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = ['-0.28*x[0]*x[2]', '0.28*x[0]*x[2] - 0.47*x[1]', '0.47*x[1] - 0.3*x[2]', '0.3*x[2]']

    def np_eq(self, t, x):
        return np.array([-0.28 * x[0] * x[2],
                         0.28 * x[0] * x[2] - 0.47 * x[1],
                         0.47 * x[1] - 0.3 * x[2],
                         0.3 * x[2]])


@register_eq_class
class PredatorPreyModel(KnownEquation):
    _eq_name = 'vars4_prog3'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "A predator-prey model with two prey and two predator species."

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 100.0), only_positive=True),
                                     LogUniformSampling((1e-2, 50.0), only_positive=True),
                                     LogUniformSampling((1e-2, 100.0), only_positive=True),
                                     LogUniformSampling((1e-2, 50.0), only_positive=True)]

        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = [
            'x[0] * (1.5 - 0.1 * x[0] - 0.05 * x[1] - 0.02 * x[2])',
            'x[1] * (-1.0 + 0.1 * x[0] - 0.07 * x[3] - 0.04 * x[1])',
            'x[2] * (1.2 - 0.08 * x[2] - 0.04 * x[3] - 0.03 * x[0])',
            'x[3] * (-0.8 + 0.09 * x[2] - 0.06 * x[1] - 0.05 * x[3])'
        ]

    def np_eq(self, t, x):
        return np.array([
            x[0] * (1.5 - 0.1 * x[0] - 0.05 * x[1] - 0.02 * x[2]),
            x[1] * (-1.0 + 0.1 * x[0] - 0.07 * x[3] - 0.04 * x[1]),
            x[2] * (1.2 - 0.08 * x[2] - 0.04 * x[3] - 0.03 * x[0]),
            x[3] * (-0.8 + 0.09 * x[2] - 0.06 * x[1] - 0.05 * x[3])
        ])


@register_eq_class
class LotkaVolterraModel(KnownEquation):
    _eq_name = 'vars4_prog4'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Lotka-Volterra equations for four competing species."

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 100.0), only_positive=True),
                                     LogUniformSampling((1e-2, 100.0), only_positive=True),
                                     LogUniformSampling((1e-2, 100.0), only_positive=True),
                                     LogUniformSampling((1e-2, 100.0), only_positive=True)]

        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = [
            'x[0] * (1.0 - 0.1 * x[0] - 0.01 * x[1] - 0.02 * x[2] - 0.01 * x[3])',
            'x[1] * (1.2 - 0.01 * x[0] - 0.1 * x[1] - 0.02 * x[2] - 0.01 * x[3])',
            'x[2] * (0.8 - 0.02 * x[0] - 0.02 * x[1] - 0.1 * x[2] - 0.01 * x[3])',
            'x[3] * (1.1 - 0.01 * x[0] - 0.01 * x[1] - 0.01 * x[2] - 0.1 * x[3])'
        ]

    def np_eq(self, t, x):
        return np.array([
            x[0] * (1.0 - 0.1 * x[0] - 0.01 * x[1] - 0.02 * x[2] - 0.01 * x[3]),
            x[1] * (1.2 - 0.01 * x[0] - 0.1 * x[1] - 0.02 * x[2] - 0.01 * x[3]),
            x[2] * (0.8 - 0.02 * x[0] - 0.02 * x[1] - 0.1 * x[2] - 0.01 * x[3]),
            x[3] * (1.1 - 0.01 * x[0] - 0.01 * x[1] - 0.01 * x[2] - 0.1 * x[3])
        ])


@register_eq_class
class ChemicalReactionNetwork(KnownEquation):
    _eq_name = 'vars4_prog5'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "A chemical reaction network involving four chemical species."

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True),
                                     LogUniformSampling((1e-2, 10.0), only_positive=True)]

        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = [
            -0.5 * x[0] + 0.2 * x[1] - 0.1 * x[0] * x[2],
            0.5 * x[0] - 0.3 * x[1] + 0.4 * x[3],
            -0.1 * x[0] * x[2] + 0.3 * x[3],
            0.1 * x[0] * x[2] - 0.4 * x[3]
        ]

    def np_eq(self, t, x):
        return np.array([
            -0.5 * x[0] + 0.2 * x[1] - 0.1 * x[0] * x[2],
            0.5 * x[0] - 0.3 * x[1] + 0.4 * x[3],
            -0.1 * x[0] * x[2] + 0.3 * x[3],
            0.1 * x[0] * x[2] - 0.4 * x[3]
        ])





@register_eq_class
class SIRModelTwoStrains(KnownEquation):
    _eq_name = 'vars4_prog6'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "SIR model with two pathogen strains."

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 1.0), only_positive=True),
                                     LogUniformSampling((1e-2, 1.0), only_positive=True),
                                     LogUniformSampling((1e-2, 1.0), only_positive=True),
                                     LogUniformSampling((1e-2, 1.0), only_positive=True)]

        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x = self.x
        self.sympy_eq = [
            -0.3 * x[0] * x[1] - 0.2 * x[0] * x[2],
            0.3 * x[0] * x[1] - 0.1 * x[1],
            0.2 * x[0] * x[2] - 0.1 * x[2],
            0.1 * x[1] + 0.1 * x[2]
        ]

    def np_eq(self, t, x):
        return np.array([
            -0.3 * x[0] * x[1] - 0.2 * x[0] * x[2],
            0.3 * x[0] * x[1] - 0.1 * x[1],
            0.2 * x[0] * x[2] - 0.1 * x[2],
            0.1 * x[1] + 0.1 * x[2]
        ])


@register_eq_class
class CoupledOscillators(KnownEquation):
    _eq_name = 'vars4_prog7'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Describes the dynamics of two coupled oscillators."

    def __init__(self):
        self.vars_range_and_types = [UniformSampling((-10, 1.0), only_positive=True),
                                     UniformSampling((-10, 1.0), only_positive=True),
                                     UniformSampling((-10, 1.0), only_positive=True),
                                     UniformSampling((-10, 1.0), only_positive=True)]
        # self.vars_range_and_types = {'x1': (-10, 10), 'v1': (-10, 10), 'x2': (-10, 10), 'v2': (-10, 10)}
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x1, v1, x2, v2 = self.x
        self.sympy_eq = {'dx1dt': v1,
                         'dv1dt': -2*x1 + 1.5*(x2 - x1),
                         'dx2dt': v2,
                         'dv2dt': -1.5*(x2 - x1) - 3*x2}

    def np_eq(self, t, x):
        return np.array([x[1],
                         -2*x[0] + 1.5*(x[2] - x[0]),
                         x[3],
                         -1.5*(x[2] - x[0]) - 3*x[2]])

@register_eq_class
class FourSpeciesCompartmentalModel(KnownEquation):
    _eq_name = 'vars4_prog8'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Describes the dynamics of four species in a compartmental model."

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 100.0), only_positive=True),
                                     LogUniformSampling((1e-2, 100.0), only_positive=True),
                                     LogUniformSampling((1e-2, 100.0), only_positive=True),
                                     LogUniformSampling((1e-2, 100.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        self.sympy_eq = ['dx1dt = -0.1*x1 + 0.05*x2',
                         'dx2dt = 0.1*x1 - (0.05 + 0.03)*x2 + 0.02*x3',
                         'dx3dt = 0.03*x2 - (0.02 + 0.01)*x3 + 0.01*x4',
                         'dx4dt = 0.01*x3 - 0.01*x4']

    def np_eq(self, t, x):
        return np.array([-0.1*x[0] + 0.05*x[1],
                         0.1*x[0] - (0.05 + 0.03)*x[1] + 0.02*x[2],
                         0.03*x[1] - (0.02 + 0.01)*x[2] + 0.01*x[3],
                         0.01*x[2] - 0.01*x[3]])
@register_eq_class
class PharmacokineticModel(KnownEquation):
    _eq_name = 'vars4_prog9'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Describes the pharmacokinetics of a drug across four compartments."

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 100.0), only_positive=True),
                                     LogUniformSampling((1e-2, 100.0), only_positive=True),
                                     LogUniformSampling((1e-2, 100.0), only_positive=True),
                                     LogUniformSampling((1e-2, 100.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        self.sympy_eq = [
            'dC1dt = -0.1*C1 + 0.05*C2',
            'dC2dt = 0.1*C1 - (0.05 + 0.03)*C2 + 0.02*C3',
            'dC3dt = 0.03*C2 - (0.02 + 0.01)*C3 + 0.01*C4',
            'dC4dt = 0.01*C3 - 0.01*C4'
        ]

    def np_eq(self, t, x):
        return np.array([
            -0.1*x[0] + 0.05*x[1],
            0.1*x[0] - (0.05 + 0.03)*x[1] + 0.02*x[2],
            0.03*x[1] - (0.02 + 0.01)*x[2] + 0.01*x[3],
            0.01*x[2] - 0.01*x[3]
        ])



@register_eq_class
class SEIRModel(KnownEquation):
    _eq_name = 'vars4_prog10'
    _operator_set = ['add', 'sub', 'mul', 'const']
    _description = "Describes the SEIR model for infectious disease spread."

    def __init__(self):
        self.vars_range_and_types = [LogUniformSampling((1e-2, 1000.0), only_positive=True),
                                     LogUniformSampling((1e-2, 1000.0), only_positive=True),
                                     LogUniformSampling((1e-2, 1000.0), only_positive=True),
                                     LogUniformSampling((1e-2, 1000.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        self.sympy_eq = ['dSdt = -0.2*S*I',
                         'dEdt = 0.2*S*I - 0.1*E',
                         'dIdt = 0.1*E - 0.1*I',
                         'dRdt = 0.1*I']

    def np_eq(self, t, x):
        return np.array([-0.2*x[0]*x[2],
                         0.2*x[0]*x[2] - 0.1*x[1],
                         0.1*x[1] - 0.1*x[2],
                         0.1*x[2]])


