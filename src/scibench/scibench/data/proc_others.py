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


def extract(row, function_set_dict):
    name = row['name'].replace('-', "_")
    nvars = row['variables']
    sympy_eq = row['expression'].replace('x10', 'x[9]').replace('x1', 'x[0]').replace('x2', 'x[1]').replace('x3', 'x[2]').replace(
        'x4', 'x[3]').replace('x5', 'x[4]').replace('x6', 'x[5]').replace('x7', 'x[6]').replace('x8', 'x[7]').replace(
        'x9', 'x[8]').replace('pow', 'sympy.Pow').replace('log', 'sympy.log').replace('exp', 'sympy.exp').replace('sin',
                                                                                                                  'sympy.sin').replace(
        'cos', 'sympy.cos').replace(
        'div', 'sympy.div').replace('sqrt', 'sympy.sqrt').replace('pi', 'sympy.pi').replace(
        'harmonic', 'sympy.harmonic')

    function_set = function_set_dict[row['function_set']]

    return template.format(name, name, function_set, nvars, sympy_eq)


@click.command()
@click.option('--benchmark_file', default="./dso_benchmarks.csv")
@click.option('--function_set_file', default="dso_function_sets.csv")
@click.option('--output_file', default="./equations_others.py")
def main(benchmark_file, function_set_file, output_file):
    all_bench_equations = pd.read_csv(benchmark_file)
    function_sets = pd.read_csv(function_set_file)
    function_set_dict = {'None': ['sqrt', 'add', 'sub', "mul", "div", "inv", 'sin', 'pow', "const"]}
    for index, row in function_sets.iterrows():
        function_set_dict[row['name']] = row['function_set'].split(',')

    # all_equations = all_bench_equations.apply(lambda row: extract(row, function_set_dict), axis=1)
    all_equations = []
    for index, row in all_bench_equations.iterrows():
        print(row['name'])
        all_equations.append(extract(row, function_set_dict))
    fw = open(os.path.join(output_file), 'w')
    fw.write(prefix)
    for line in all_equations:
        fw.write(line + '\n')


if __name__ == '__main__':
    main()
