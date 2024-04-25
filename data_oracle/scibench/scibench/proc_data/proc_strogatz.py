import pandas as pd
import os
from sympy import Symbol, parse_expr, simplify

prefix = """import numpy as np
from scibench.data.base import KnownEquation, register_eq_class
from scibench.symbolic_data_generator import LogUniformSampling
"""

template = """@register_eq_class
class {}(KnownEquation):
    _eq_name = '{}'
    _operator_set = {}
    # description: {}
    def __init__(self):
        
        self.vars_range_and_types = [{}]
        super().__init__(num_vars={}, vars_range_and_types=self.vars_range_and_types)
        
    
    def np_eq(self, t, x):
        return np.array([{}])
"""


def fill_template(expressions, class_name, name, nvars, description, function_set):
    element = 'LogUniformSampling((1e-2, 10.0), only_positive=True)'
    elements = ", ".join([element for _ in range(int(nvars))])
    return template.format(class_name, name, function_set, description, elements, nvars, ", ".join(expressions))


def detect_function_set(one_ode):
    function_set_base = ['add', 'sub', 'mul', 'div', 'const']
    if 'sin' in one_ode:
        function_set_base.append('sin')
    if 'cos' in one_ode:
        function_set_base.append('cos')
    if 'exp' in one_ode:
        function_set_base.append('exp')
    if 'log' in one_ode:
        function_set_base.append('log')
    if 'cot' in one_ode:
        function_set_base.append('cot')
    if 'abs' in one_ode or 'Abs' in one_ode > 0:
        function_set_base.append('abs')
    return function_set_base


def main():
    from strogatz_equations import equations
    fw = open(os.path.join("strogatz.py"), 'w')
    fw.write(prefix)
    idx_dicts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    for one_eq in equations:
        print(one_eq)
        expressions = one_eq['eq']

        consts = one_eq['consts'][0]
        x = [Symbol(f'x_{i}') for i in range(one_eq['dim'])]

        expressions = expressions.replace('^', '**')
        for i in range(len(consts)):
            expressions = expressions.replace(f'c_{i}', str(consts[i]))
        expressions = expressions.split(' | ')
        expressions = [str(simplify(parse_expr(eq).expand())) for eq in expressions]
        expressions = " | ".join(expressions)
        expressions = expressions.replace('exp', 'np.exp')
        expressions = expressions.replace('log', 'np.log')
        expressions = expressions.replace('sin', 'np.sin')
        expressions = expressions.replace('cos', 'np.cos')
        expressions = expressions.replace('cot', 'np.cot')
        expressions = expressions.replace('abs', 'np.abs')
        expressions = expressions.replace('Abs', 'np.abs')
        function_set = detect_function_set(expressions)
        for i in range(one_eq['dim']):
            if f'x_{i}' in expressions:
                expressions = expressions.replace(f'x_{i}', f'x[{i}]')
        expressions = expressions.split(' | ')
        description = one_eq['eq_description']
        idx_dicts[one_eq['dim']] += 1
        name = "vars{}_prog{}".format(one_eq['dim'], idx_dicts[one_eq['dim']])

        class_name = one_eq['source']
        class_name = class_name.replace('.', "_")
        class_name = class_name.replace(' ', '_')
        class_name = class_name.replace('-', '_')
        class_name = class_name.upper()

        line = fill_template(expressions, class_name, name, one_eq['dim'], description, function_set)

        fw.write(line + '\n')
    fw.close()
    print(idx_dicts)


if __name__ == '__main__':
    main()
