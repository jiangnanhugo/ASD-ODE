"""optimize the coefficients in the candidate ODEs"""
import sys

import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.set_printoptions(precision=4, linewidth=np.inf)

from grammar.utils import pretty_print_expr
from grammar.production_rules import check_non_terminal_nodes

from sympy.parsing.sympy_parser import parse_expr
from sympy import lambdify, symbols
from scipy.optimize import minimize
from scipy.optimize import basinhopping, shgo, dual_annealing, direct
from scipy.integrate import solve_ivp

# from grammar.odeint.numpy_odeint import runge_kutta4


def optimize(candidate_ode_equations: list, init_cond, time_span, t_eval, true_trajectories, input_var_Xs,
             loss_func, max_open_constants, max_opt_iter,
             optimizer_name,
             non_terminal_nodes,
             user_scpeficied_iters=-1,
             verbose=False):
    """
    optimize the constant coefficients in the candidate expressions.

    candidate_ode_equations: list of strings for n ODE expressions. the discovered equation
                             (with placeholders for coefficients).
    init_cond: [batch_size, nvars]. the initial conditions of each variables.
    true_trajectories: [batch_size, time_steps, nvars]. the true trajectories.
    # other parameters:
    max_open_constants: the maximum opening constant allowed for each ODE.
    input_var_Xs: list of sympy.symbol object. It used to construct the scipy.lambda function for the candidate expressions.
    This optimize function does not solve expressions with too many coefficients, because it is too slow.

    loss_func: the loss function for computing the distance between the true trajectories and predicted trajectories.

    max_opt_iter: maximum number of optimization iterations.
    user_scpeficied_iters: user specified number of optimization iterations.

    optimizer_name: name of the optimizer. See scipy.optimize.minimize for list of optimizers
    non_terminal_nodes: list of non-terminal nodes. It is used for checking if the expression is valid

    """

    candidate_ode_equations = simplify_template(candidate_ode_equations)
    print("candidate:", candidate_ode_equations)
    if check_non_terminal_nodes(candidate_ode_equations, non_terminal_nodes):
        # not a valid equation
        return -np.inf, candidate_ode_equations, 0, np.inf

    # count the total number of constants in equation
    num_changing_consts = sum([x.count('C') for x in candidate_ode_equations])
    t_optimized_constants, t_optimized_obj = 0, np.inf
    if num_changing_consts == 0:
        # zero constant
        var_ytrue = np.var(true_trajectories)
        pred_trajectories = execute(candidate_ode_equations, init_cond, time_span, t_eval, input_var_Xs)
    elif num_changing_consts >= max_open_constants:
        # discourage over expressions with too many coefficients.
        return -np.inf, candidate_ode_equations, t_optimized_constants, t_optimized_obj
    else:
        c_lst = ['c' + str(i) for i in range(num_changing_consts)]
        temp_equations = "$$".join(candidate_ode_equations)
        for c in c_lst:
            temp_equations = temp_equations.replace('C', c, 1)
        candidate_ode_equations = temp_equations.split("$$")

        def f(consts):
            temp_equations = "$$".join(candidate_ode_equations)
            for i in range(len(consts)):
                temp_equations = temp_equations.replace('c' + str(i), str(consts[i]), 1)
            eq_est = temp_equations.split("$$")
            pred_trajectories = execute(eq_est, init_cond, time_span, t_eval, input_var_Xs)
            var_ytrue = np.var(true_trajectories)
            loss_val = -loss_func(pred_trajectories, true_trajectories, var_ytrue)
            return loss_val

        # do more than one experiment,
        x0 = np.random.rand(len(c_lst))
        try:
            if user_scpeficied_iters > 0:
                max_opt_iter = user_scpeficied_iters
            opt_result = scipy_minimize(f, x0, optimizer_name, num_changing_consts, max_opt_iter)
            t_optimized_constants = opt_result['x']
            c_lst = t_optimized_constants.tolist()
            t_optimized_obj = opt_result['fun']

            if verbose:
                print(opt_result)
            eq_est = candidate_ode_equations

            for i in range(len(c_lst)):
                temp = []
                for one_eq in eq_est:
                    est_c = np.mean(c_lst[i])
                    if abs(est_c) < 1e-5:
                        est_c = 0
                    one_eq = one_eq.replace('c' + str(i), str(est_c), 1)
                    temp.append(one_eq)
                eq_est = temp

            pred_trajectories = execute(eq_est, init_cond, time_span, t_eval, input_var_Xs)
            # what is this?
            var_ytrue = np.var(true_trajectories)

            candidate_ode_equations = [pretty_print_expr(parse_expr(one_expr)) for one_expr in eq_est]
        except Exception as e:
            print(e)
            return -np.inf, candidate_ode_equations, 0, np.inf

    # r = eta ** tree_size * float(-np.log10(1e-60 - self.loss_func(pred_trajectories, y_true, var_ytrue)))
    if pred_trajectories is None:
        return -np.inf, candidate_ode_equations, 0, np.inf

    train_loss = loss_func(pred_trajectories, true_trajectories, var_ytrue)
    print('\t metric:', train_loss, 'eq:', candidate_ode_equations)
    sys.stdout.flush()
    return train_loss, candidate_ode_equations, t_optimized_constants, t_optimized_obj


def scipy_minimize(f, x0, optimizer, num_changing_consts, max_opt_iter):
    # optimize the open constants in the expression
    opt_result = None
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


def execute(expr_strs: list[str], x_init_conds: np.ndarray, time_span: tuple, t_evals: np.ndarray,
            input_var_Xs: list) -> np.ndarray:
    """
    given a symbolic ODE (func) and the initial condition (init_cond), compute the time trajectory.

    expr_strs: list of string. each string is one expression.
    time_span: tuple
    t_evals: np.linspace, or np.logspace
    x_init_conds: [batch_size, nvars]
    pred_trajectories: [batch_size, time_steps, nvars]
    """
    # sttime=time.time()

    expr_odes = [parse_expr(one_expr) for one_expr in expr_strs]
    t = symbols('t')  # not used in this case
    try:
        func = lambdify((t, input_var_Xs), expr_odes, modules='numpy')
        pred_trajectories = []
        for one_x_init in x_init_conds:
            # one_solution = runge_kutta4(func, t_evals, one_x_init)
            one_solution = solve_ivp(func, tspan=time_span, t_evals=t_evals, y0=one_x_init, method='RK45')
            pred_trajectories.append(one_solution.y)
        pred_trajectories = np.asarray(pred_trajectories)

        if pred_trajectories is complex:
            pred_trajectories = np.ones(x_init_conds.shape[0], t_evals.shape[0], x_init_conds.shape[1]) * -np.infty
            return pred_trajectories
            # return np.ones(init_cond.shape[-1]) * np.infty
    except TypeError as e:
        # print(e, expr, input_var_Xs, data_X.shape)
        pred_trajectories = np.ones(x_init_conds.shape[0], t_evals.shape[0], x_init_conds.shape[1]) * -np.infty
        return pred_trajectories
    except KeyError as e:
        # print(e, expr)
        pred_trajectories = np.ones(x_init_conds.shape[0], t_evals.shape[0], x_init_conds.shape[1]) * -np.infty
        return pred_trajectories
    except ValueError as e:
        pred_trajectories = np.ones(x_init_conds.shape[0], t_evals.shape[0], x_init_conds.shape[1]) * -np.infty
        return pred_trajectories
    return pred_trajectories


def simplify_template(equations: list) -> list:
    new_equations = []
    for eq in equations:
        for i in range(10):
            eq = eq.replace('(C+C)', 'C')
            eq = eq.replace('(C-C)', 'C')
            #
            eq = eq.replace('C*C', 'C')
            eq = eq.replace('(C)*C', 'C')
            eq = eq.replace('C*(C)', 'C')
            eq = eq.replace('(C)*(C)', 'C')
            #
            eq = eq.replace('C/C', 'C')
            eq = eq.replace('(C)/C', 'C')
            eq = eq.replace('C/(C)', 'C')
            eq = eq.replace('(C)/(C)', 'C')
            eq = eq.replace('sqrt(C)', 'C')
            eq = eq.replace('exp(C)', 'C')
            eq = eq.replace('log(C)', 'C')
            eq = eq.replace('sin(C)', 'C')
            eq = eq.replace('cos(C)', 'C')
            eq = eq.replace('(1/C)', 'C')
        new_equations.append(eq)
    return new_equations
