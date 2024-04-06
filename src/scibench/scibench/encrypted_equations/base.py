from sympy import Symbol
import json


class KnownEquation(object):
    _eq_name = None
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'pow', 'sin', 'cos', 'exp', 'log', 'const']
    simulated_exec = False

    def __init__(self, num_vars, vars_range_and_types=None, kwargs_list=None):
        if kwargs_list is None:
            kwargs_list = [{'real': True} for _ in range(num_vars)]

        assert len(kwargs_list) == num_vars
        self.num_vars = num_vars
        self.vars_range_and_types = vars_range_and_types
        self.x = [Symbol(f'X{i}', **kwargs) for i, kwargs in enumerate(kwargs_list)]
        # the dimension of each variables is 1 by default.
        self.dim = [(1) for i in range(self.num_vars)]
        self.sympy_eq = None

    def vars_range_and_types_to_json_str(self):
        if self.vars_range_and_types:
            return json.dumps([one.to_dict() for one in self.vars_range_and_types])
        else:
            default = {'name': 'LogUniform',
                       'range': [0.1, 10],
                       'dim': [1, ],
                       'only_positive': True}
            return json.dumps([default for _ in range(self.num_vars)])


class DefaultSampling(object):
    def __init__(self, name, min_value, max_value, only_positive=False, dim=(1,)):
        """

        Parameters
        ----------
        name: name of the sampling algorithms
        min_value
        max_value
        only_positive
        dim: dimension of the input variable.
        """
        self.name = name
        self.range = [min_value, max_value]
        self.only_positive = only_positive
        self.dim = dim

    def to_dict(self):
        return {'name': self.name,
                'range': self.range,
                'dim': self.dim,
                'only_positive': self.only_positive}


class LogUniformSampling(DefaultSampling):
    def __init__(self, min_value, max_value, only_positive=False, dim=(1,)):
        super().__init__('LogUniform', min_value, max_value, only_positive, dim=dim)


class IntegerUniformSampling(DefaultSampling):
    def __init__(self, min_value, max_value, only_positive=False, dim=(1,)):
        super().__init__('IntegerUniform', int(min_value), int(max_value), only_positive, dim=dim)


class UniformSampling(DefaultSampling):
    def __init__(self, min_value, max_value, only_positive=False, dim=(1,)):
        super().__init__('Uniform', min_value, max_value, only_positive, dim=dim)


# class DefaultSampling2d(object):
#     def __init__(self, name, min_value, max_value, only_positive=False, dim=(1, 1)):
#         self.name = name
#         self.range = range
#         self.only_positive = only_positive
#         self.dim = dim


class LogUniformSampling2d(DefaultSampling):
    def __init__(self, min_value, max_value, only_positive=False, dim=(1, 1)):
        super().__init__('LogUniform2d', min_value, max_value, only_positive, dim=dim)
