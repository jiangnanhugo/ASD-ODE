"""Class for symbolic expression optimization."""

import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.set_printoptions(precision=4, linewidth=np.inf)
from sympy.parsing.sympy_parser import parse_expr
from sympy import lambdify

from scipy.optimize import minimize
from scipy.optimize import basinhopping, shgo, dual_annealing, direct
from grammar import pretty_print_expr
from grammar.production_rules import concate_production_rules_to_expr
from grammar import all_metrics
from pathos.multiprocessing import ProcessPool
from itertools import chain
from scipy.integrate import solve_ivp
import sys


class SymbolicDifferentialEquations(object):
    """
    this class is used to represent symbolic ordinary differential equations.
    For n variables settings, there will be n total expressions.
    """
    def __init__(self, list_of_rules):
        self.traversal = list_of_rules
        self.expr_template = concate_production_rules_to_expr(list_of_rules)
        self.reward = None
        self.fitted_eq = None
        self.invalid = False
        self.all_metrics = None

    def __repr__(self):
        return f"r={self.reward}, eq={self.fitted_eq}"

    def print_all_metrics(self):
        print('-' * 30)
        if not self.all_metrics:
            print("No metrics")
            print('-' * 30)
            return
        for mertic_name in self.all_metrics:
            print(f"{mertic_name} {self.all_metrics[mertic_name]}")
        print('-' * 30)


class grammarProgram(object):
    """
    used for optimizing the constants in the expressions.
    """
    evaluate_loss = None

    def __init__(self, optimizer="BFGS", metric_name='inv_nrmse', max_opt_iter=100, n_cores=1, max_open_constants=20):
        """
        opt_num_expr:  # number of experiments done for optimization
        max_open_constants: the maximum number of allowed open constants in the expression.
        """

        self.optimizer = optimizer
        self.max_opt_iter = max_opt_iter
        self.max_open_constants = max_open_constants
        self.metric_name = metric_name
        self.n_cores = n_cores
        self.evaluate_loss = all_metrics[metric_name]
        self.pool = ProcessPool(nodes=self.n_cores)

    def fitting_new_expressions(self, many_seqs_of_rules, dataX: np.ndarray, y_true, input_var_Xs):
        """
        here we assume the input will be a valid expression
        """
        result = []
        print("many_seqs_of_rules:", len(many_seqs_of_rules))
        for one_list_rules in many_seqs_of_rules:
            one_expr = SymbolicDifferentialEquations(one_list_rules)
            reward, fitted_eq, _, _ = optimize(
                one_expr.expr_template,
                dataX,
                y_true,
                input_var_Xs,
                self.evaluate_loss,
                self.max_open_constants,
                self.max_opt_iter,
                self.optimizer
            )

            one_expr.reward = reward
            one_expr.fitted_eq = fitted_eq
            result.append(one_expr)
        return result

    def fitting_new_expressions_in_parallel(self, many_seqs_of_rules, dataX: np.ndarray, y_true, input_var_Xs):
        """
        here we assume the input will be a valid expression
        """

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            chunk_size = len(lst) // n
            for i in range(0, len(lst), chunk_size):
                yield lst[i:i + chunk_size]

        many_expr_tempaltes = chunks(
            [SymbolicDifferentialEquations(one_rules) for one_rules in many_seqs_of_rules],
            self.n_cores)
        dataXes = [dataX for _ in range(self.n_cores)]
        y_trues = [y_true for _ in range(self.n_cores)]
        input_var_Xes = [input_var_Xs for _ in range(self.n_cores)]
        evaluate_losses = [self.evaluate_loss for _ in range(self.n_cores)]
        max_open_constantes = [self.max_open_constants for _ in range(self.n_cores)]
        max_opt_iteres = [self.max_opt_iter for _ in range(self.n_cores)]
        optimizeres = [self.optimizer for _ in range(self.n_cores)]

        result = self.pool.map(fit_one_expr, many_expr_tempaltes, dataXes, y_trues, input_var_Xes, evaluate_losses,
                               max_open_constantes, max_opt_iteres, optimizeres)
        result = list(chain.from_iterable(result))
        print("Done with optimization!")
        sys.stdout.flush()
        return result


def fit_one_expr(one_expr_batch, dataX, y_true, input_var_Xs, evaluate_loss, max_open_constants, max_opt_iter,
                 optimizer_name):
    results = []
    for one_expr in one_expr_batch:
        reward, fitted_eq, _, _ = optimize(one_expr.expr_template,
                                           dataX,
                                           y_true,
                                           input_var_Xs,
                                           evaluate_loss, max_open_constants, max_opt_iter, optimizer_name)

        one_expr.reward = reward
        one_expr.fitted_eq = fitted_eq
        results.append(one_expr)
    return results


def optimize(eq, data_X, y_true, input_var_Xs, evaluate_loss, max_open_constants, max_opt_iter, optimizer_name,
             user_scpeficied_iters=-1,
             verbose=False):
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
        return -np.inf, eq, 0, np.inf

    # count number of constants in equation
    num_changing_consts = eq.count('C')
    t_optimized_constants, t_optimized_obj = 0, np.inf
    if num_changing_consts == 0:  # zero constant
        var_ytrue = np.var(y_true)
        y_pred = execute(eq, data_X.T, input_var_Xs)
    elif num_changing_consts >= max_open_constants:  # discourage over complicated numerical estimations
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
            loss_val = -evaluate_loss(y_pred, y_true, var_ytrue)
            return loss_val

        # do more than one experiment,
        x0 = np.random.rand(len(c_lst))
        try:
            max_iter = max_opt_iter
            if user_scpeficied_iters > 0:
                max_iter = user_scpeficied_iters
            opt_result = scipy_minimize(f, x0, optimizer_name, num_changing_consts, max_iter)
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

            print('\t loss:', -evaluate_loss(y_pred, y_true, var_ytrue), '\t fitted eq:', eq)
        except Exception as e:
            print(e)
            return -np.inf, eq, 0, np.inf

    # r = eta ** tree_size * float(-np.log10(1e-60 - self.evaluate_loss(y_pred, y_true, var_ytrue)))
    reward = evaluate_loss(y_pred, y_true, var_ytrue)

    return reward, eq, t_optimized_constants, t_optimized_obj


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
        if len(used_idx) == 0:
            return float(expr)
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


def scipy_minimize(f, x0, optimizer, num_changing_consts, max_opt_iter):
    # optimize the open constants in the expression
    if optimizer == 'Nelder-Mead':
        opt_result = minimize(f, x0, method='Nelder-Mead',
                              options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': max_opt_iter})
    elif optimizer == 'BFGS':
        opt_result = minimize(f, x0, method='BFGS', options={'maxiter': max_opt_iter})
    elif optimizer == 'CG':
        opt_result = minimize(f, x0, method='CG', options={'maxiter': max_opt_iter})
    elif optimizer == 'L-BFGS-B':
        opt_result = minimize(f, x0, method='L-BFGS-B', options={'maxiter': max_opt_iter})
    elif optimizer == "basinhopping":
        minimizer_kwargs = {"method": "Nelder-Mead",
                            "options": {'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 100}}
        opt_result = basinhopping(f, x0, minimizer_kwargs=minimizer_kwargs, niter=max_opt_iter)
    elif optimizer == 'dual_annealing':
        minimizer_kwargs = {"method": "Nelder-Mead",
                            "options": {'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 100}}
        lw = [-5] * num_changing_consts
        up = [5] * num_changing_consts
        bounds = list(zip(lw, up))
        opt_result = dual_annealing(f, bounds, minimizer_kwargs=minimizer_kwargs, maxiter=max_opt_iter)
    elif optimizer == 'shgo':
        minimizer_kwargs = {"method": "Nelder-Mead",
                            "options": {'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 100}}
        lw = [-5] * num_changing_consts
        up = [5] * num_changing_consts
        bounds = list(zip(lw, up))
        opt_result = shgo(f, bounds, minimizer_kwargs=minimizer_kwargs, options={'maxiter': max_opt_iter})
    elif optimizer == "direct":
        lw = [-10] * num_changing_consts
        up = [10] * num_changing_consts
        bounds = list(zip(lw, up))
        opt_result = direct(f, bounds, maxiter=max_opt_iter)

    return opt_result

def compute_time_trajectory(func, init_cond, t_span, t_eval=np.linspace(0, 10, 50) ):
    """
    given a symbolic ODE (func) and the initial condition (init_cond), compute the time trajectory.
    https://docs.sympy.org/latest/guides/solving/solve-ode.html
    """
    solution =  solve_ivp(func, t_span=t_span, y0=init_cond, t_eval=t_eval)
    # Extract the y (concentration) values from SciPy solution result
    return solution.y

def simplify_template(eq):
    for i in range(10):
        eq = eq.replace('(C+C)', 'C')
        eq = eq.replace('(C-C)', 'C')
        eq = eq.replace('C*C', 'C')
        eq = eq.replace('(/C', 'C')
        eq = eq.replace('sqrt(C)', 'C')
        eq = eq.replace('exp(C)', 'C')
        eq = eq.replace('log(C)', 'C')
        eq = eq.replace('sin(C)', 'C')
        eq = eq.replace('cos(C)', 'C')
        eq = eq.replace('(1/C)', 'C')
    return eq


if __name__ == '__main__':
    expr_temp = 'sqrt(sqrt(C))*(sqrt(X0)+C)'

    simplify_template(expr_temp)
