import numpy as np

from ProGED.equation_discoverer import EqDisco
from ProGED.generators.load_generator import LoadGenerator


if __name__ == '__main__':
    num_points = 1000
    x = np.random.random(10000) * 20 - 10
    y = x ** 2 / 8 - x + 12
    estimated = 0.124999999496695 * x * (x - 8.00000003873915) + 12.0000000191275
    mat = np.stack([x, y]).T
    generator = LoadGenerator("data/example_eqs.txt")
    ed = EqDisco(data=mat,  lhs_vars=["y"],
                 # list the variable on the left-hand side of each equation in the system of equations, should match columns names in data
                 rhs_vars=['x'], generator=generator, sample_size=100)
    ed.generate_models()
    print(ed.models)
    ed.fit_models()
    print(ed.get_results())
