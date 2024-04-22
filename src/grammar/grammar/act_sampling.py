# given a set of ODEs expressions,
# 1. find the fixed point of each ODE,
# 2. determine the type of each fixed point by Jacobain
# 3. determine some regions where most ODEs disagreee
# 4. sample some initial condition from these regions.
from scipy.optimize import fsolve
import numpy as np

from sympy import Matrix, Symbol, nonlinsolve


def find_fixed_points(list_of_one_ode: list, input_var_Xs: list, type=0):
    list_of_fixed_points = []
    for one_ode in list_of_one_ode:
        # https://docs.sympy.org/latest/modules/solvers/solveset.html#sympy.solvers.solveset.nonlinsolve
        fixed_points = nonlinsolve(one_ode, input_var_Xs)
        list_of_fixed_points.append(fixed_points)
    return fixed_points




def pipeline(one_ode: list, input_var_Xs: list):
    """

    https://stackoverflow.com/questions/49553006/compute-the-jacobian-matrix-in-python
    example:
    Matrix(['2*u1 + 3*u2','2*u1 - 3*u2']).jacobian(['u1', 'u2'])
    """
    # 1. find_fixed_points
    # https://docs.sympy.org/latest/modules/solvers/solveset.html#sympy.solvers.solveset.nonlinsolve
    fixed_points = nonlinsolve(one_ode, input_var_Xs)
    # 2. compute_jacobian
    Y = Matrix(one_ode)
    jacob = Y.jacobian(input_var_Xs)
    # determine the type of each fixed_points
    for one_point in fixed_points:
        temp_matrix = jacob.evalf(subs={one_point: one_point})
