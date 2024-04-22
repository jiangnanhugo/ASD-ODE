# given a set of ODEs expressions,
# 1. find the fixed point of each ODE,
# 2. determine the type of each fixed point by Jacobain
# 3. determine some regions where most ODEs disagreee
# 4. sample some initial condition from these regions.
from scipy.optimize import fsolve
import numpy as np
def root_solve(one_ode):
    init_guess=np.random.randn(one_ode.shape[0])
    root = fsolve(one_ode, init_guess)



def compute_jacobian(one_ode):
    """
    https://stackoverflow.com/questions/49553006/compute-the-jacobian-matrix-in-python
    example:
    Matrix(['2*u1 + 3*u2','2*u1 - 3*u2']).jacobian(['u1', 'u2'])
    """
    pass


