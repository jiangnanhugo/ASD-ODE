"""Class for symbolic expression object or program."""
import copy

import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.set_printoptions(precision=4, linewidth=np.inf)

from sympy.parsing.sympy_parser import parse_expr
from sympy import lambdify

from scipy.optimize import minimize
from scipy.optimize import basinhopping, shgo, dual_annealing

from utils import pretty_print_expr


class Program(object):
    # Static variables
    expr_obj_thres = 1e-6  # expression objective threshold
    expr_consts_thres = 1e-3
    evalaute_loss = None

    def __init__(self, n_vars, optimizer="BFGS"):
        """
        opt_num_expr:  # number of experiments done for optimization
        """

        self.n_vars = n_vars
        self.optimizer = optimizer

        self.optimized_constants = []
        self.optimized_obj = []
        self.cache = {}



    def optimize(self, eq, tree_size: int, data_X, y_true, input_var_Xs, eta=0.9999, max_opt_iter=1000, verbose=False):
        """
        Calculate reward score for a complete parse tree
        If placeholder C is in the equation, also execute estimation for C
        Reward = 1 / (1 + MSE) * Penalty ** num_term

        Parameters
        ----------
        eq : Str object. the discovered equation (with placeholders for coefficients).
        tree_size: number of production rules in the complete parse tree.
        (data_X, y_true) : 2-d numpy array.

        Returns
        -------
        score: discovered equations.
        eq: discovered equations with estimated numerical values.
        """
        eq = simplify_template(eq)
        if 'A' in eq or 'B' in eq:  # not a valid equation
            return -np.inf, eq, 0, 0
        # count number of constants in equation
        num_changing_consts = eq.count('C')
        t_optimized_constants, t_optimized_obj = 0, np.inf
        if num_changing_consts == 0:  # zero constant
            y_pred = execute(eq, data_X.T, input_var_Xs)
        elif num_changing_consts >= 20:  # discourage over complicated numerical estimations
            return -np.inf, eq, t_optimized_constants, t_optimized_obj
        else:
            c_lst = ['c' + str(i) for i in range(num_changing_consts)]
            for c in c_lst:
                eq = eq.replace('C', c, 1)

            def f(consts: list):
                eq_est = eq
                for i in range(len(consts)):
                    eq_est = eq_est.replace('c' + str(i), str(consts[i]), 1)
                eq_est = eq_est.replace('+ -', '-')
                eq_est = eq_est.replace('- -', '+')
                eq_est = eq_est.replace('- +', '-')
                eq_est = eq_est.replace('+ +', '+')
                y_pred = execute(eq_est, data_X.T, input_var_Xs)
                var_ytrue = np.var(y_true)
                return -self.evalaute_loss(y_pred, y_true, var_ytrue)

            # do more than one experiment,
            x0 = np.random.rand(len(c_lst))
            try:
                # optimize the constants in the expression
                if self.optimizer == 'Nelder-Mead':
                    opt_result = minimize(f, x0, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': max_opt_iter})

                elif self.optimizer == 'BFGS':
                    opt_result = minimize(f, x0, method='BFGS', options={'maxiter': max_opt_iter})
                elif self.optimizer == 'CG':
                    opt_result = minimize(f, x0, method='CG', options={'maxiter': max_opt_iter})
                elif self.optimizer == 'L-BFGS-B':
                    opt_result = minimize(f, x0, method='L-BFGS-B', options={'maxiter': max_opt_iter})
                elif self.optimizer == "basinhopping":
                    minimizer_kwargs = {"method": "Nelder-Mead",
                                        "options": {'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 100}}
                    opt_result = basinhopping(f, x0, minimizer_kwargs=minimizer_kwargs, niter=max_opt_iter)
                elif self.optimizer == 'dual_annealing':
                    minimizer_kwargs = {"method": "Nelder-Mead",
                                        "options": {'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 100}}
                    lw = [-5] * num_changing_consts
                    up = [5] * num_changing_consts
                    bounds = list(zip(lw, up))
                    opt_result = dual_annealing(f, bounds, minimizer_kwargs=minimizer_kwargs, maxiter=max_opt_iter)
                elif self.optimizer == 'shgo':
                    minimizer_kwargs = {"method": "Nelder-Mead",
                                        "options": {'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 100}}
                    lw = [-5] * num_changing_consts
                    up = [5] * num_changing_consts
                    bounds = list(zip(lw, up))
                    opt_result = shgo(f, bounds, minimizer_kwargs=minimizer_kwargs, options={'maxiter': max_opt_iter})


                t_optimized_constants = opt_result['x']
                c_lst = t_optimized_constants.tolist()
                t_optimized_obj = opt_result['fun']

                if verbose:
                    print(opt_result)
                eq_est = eq

                for i in range(len(c_lst)):
                    est_c = np.mean(c_lst[i])
                    if abs(est_c) < 1e-5:
                        est_c = 0
                    eq_est = eq_est.replace('c' + str(i), str(est_c), 1)
                eq_est = eq_est.replace('+ -', '-')
                eq_est = eq_est.replace('- -', '+')
                eq_est = eq_est.replace('- +', '-')
                eq_est = eq_est.replace('+ +', '+')

                y_pred = execute(eq_est, data_X.T, input_var_Xs)
                var_ytrue = np.var(y_true)

                eq = pretty_print_expr(parse_expr(eq_est))

                print('\t loss:', -self.evalaute_loss(y_pred, y_true, var_ytrue),
                      'Eq:', eq)
            except Exception as e:
                print(e)
                return -np.inf, eq, 0, np.inf

        r = self.evalaute_loss(y_pred, y_true, var_ytrue)

        return r, eq, t_optimized_constants, t_optimized_obj


def execute(expr_str: str, data_X: np.ndarray, input_var_Xs):
    """
    evaluate the output of expression with the given input.
    consts: list of constants.
    """
    expr = parse_expr(expr_str)
    used_vars, used_idx = [], []
    for idx, xi in enumerate(input_var_Xs):
        if str(xi) in expr_str:
            used_idx.append(idx)
            used_vars.append(xi)
    try:
        f = lambdify(used_vars, expr, 'numpy')
        if len(used_idx) != 0:
            y_hat = f(*[data_X[i] for i in used_idx])
        else:
            y_hat = float(expr)
        if y_hat is complex:
            return np.ones(data_X.shape[-1]) * np.infty
    except TypeError as e:
        # print(e, expr, input_var_Xs, data_X.shape)
        y_hat = np.ones(data_X.shape[-1]) * np.infty
    except KeyError as e:
        # print(e, expr)
        y_hat = np.ones(data_X.shape[-1]) * np.infty

    return y_hat





def simplify_template(eq):
    for i in range(10):
        eq = eq.replace('(C+C)', 'C')
        eq = eq.replace('sqrt(C)', 'C')
        eq = eq.replace('sin(C)', 'C')
        eq = eq.replace('cos(C)', 'C')
        eq = eq.replace('(1/C)', 'C')
        eq = eq.replace('(C-C)', 'C')
        eq = eq.replace('C*C', 'C')
        eq = eq.replace('(C/C)', 'C')
    return eq


if __name__ == '__main__':
    expr_temp = 'sqrt(sqrt(C))*(sqrt(X0)+C)'

    simplify_template(expr_temp)
