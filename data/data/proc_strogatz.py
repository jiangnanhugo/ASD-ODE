import pandas as pd
import os
import click

prefix = """from collections import OrderedDict
import sympy
from base import KnownEquation

EQUATION_CLASS_DICT = OrderedDict()


def register_eq_class(cls):
    EQUATION_CLASS_DICT[cls.__name__] = cls
    return cls


def get_eq_obj(key, **kwargs):
    if key in EQUATION_CLASS_DICT:
        return EQUATION_CLASS_DICT[key](**kwargs)
    raise KeyError(f'`{key}` is not expected as a equation object key')

"""

template = """@register_eq_class
class {}(KnownEquation):
    _eq_name = '{}'
    _function_set = {}

    def __init__(self):
        super().__init__(num_vars={})
        x = self.x
        self.sympy_eq = {}
"""


def detect_function_set(one_ode):
    function_set_base=['add','sub','mul','div']
    if sum(['sin' in one_eq for one_eq in one_ode])>0:
        function_set_base.append('sin')

    if sum(['cos' in one_eq for one_eq in one_ode])>0:
        function_set_base.append('exp')
    if sum(['exp' in one_eq for one_eq in one_ode])>0:
        function_set_base.append('exp')
    return function_set_base


@click.command()
@click.option('--function_set_file', default="dso_function_sets.csv")
@click.option('--output_file', default="./equations_others.py")
def main(function_set_file, output_file):
    from strogatz_equations import equations
    for one_eq in equations:
        expressions = one_eq['eq']
        for i in one_eq['dim']:
            if f'x_{i}' in expressions:
                expressions = expressions.replace(f'x_{i}', 'x[{i}]')
        expressions=expressions.split(' | ')

        description= one_eq['eq_description']
        name= "vars{}_prog{}".format(one_eq['dim'],one_eq['id'])
        coefficients = one_eq['consts'][0]
        # all_bench_equations = pd.read_csv(benchmark_file)
        # function_sets = pd.read_csv(function_set_file)
        # function_set_dict = {'None': ['sqrt', 'add', 'sub', "mul", "div", "inv", 'sin', 'pow', "const"]}
        # for index, row in function_sets.iterrows():
        #     function_set_dict[row['name']] = row['function_set'].split(',')
        #
        # # all_equations = all_bench_equations.apply(lambda row: extract(row, function_set_dict), axis=1)
        # all_equations = []
        # for index, row in all_bench_equations.iterrows():
        #     print(row['name'])
        #     all_equations.append(extract(row, function_set_dict))
        # fw = open(os.path.join(output_file), 'w')
        # fw.write(prefix)
        # for line in all_equations:
        #     fw.write(line + '\n')


if __name__ == '__main__':
    main()
