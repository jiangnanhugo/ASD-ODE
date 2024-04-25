"""Class for symbolic expression optimization."""

from sympy.parsing.sympy_parser import parse_expr
from sympy import lambdify, symbols

import numpy as np
import warnings

# from scibench.solve_init_value_problem import runge_kutta4

from scipy.optimize import minimize
from scipy.optimize import basinhopping, shgo, dual_annealing, direct
from scipy.integrate import solve_ivp

from grammar.grammar_utils import pretty_print_expr
from grammar.production_rules import check_non_terminal_nodes

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.set_printoptions(precision=4, linewidth=np.inf)


def euler_method(func, times, x_init):
    """
    https://perso.crans.org/besson/publis/notebooks/Runge-Kutta_methods_for_ODE_integration_in_Python.html
    """
    n = len(times)
    y = np.zeros((n, len(x_init)))
    y[0] = x_init
    for i in range(len(times) - 1):
        y[i + 1] = y[i] + (times[i + 1] - times[i]) * np.asarray(func(times[i], y[i]))
    return y

def runge_kutta4(func, times, x_init):
    """
    solve a batch of initial conditions
    """
    n = len(times)
    y = np.zeros((n, len(x_init)))
    y[0] = x_init
    for i in range(len(times) - 1):
        h = times[i + 1] - times[i]
        k1 = np.asarray(func(times[i], y[i]))
        k2 = np.asarray(func(times[i] + h / 2., y[i] + k1 * h / 2))
        k3 = np.asarray(func(times[i] + h / 2, y[i] + k2 * h / 2))
        k4 = np.asarray(func(times[i] + h, y[i] + k3 * h))
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def optimize(candidate_ode_equations: list, init_cond, time_span, t_eval, true_trajectories, input_var_Xs,
             loss_func, max_open_constants, max_opt_iter,
             optimizer_name,
             non_terminal_nodes,
             user_scpeficied_iters=-1,
             verbose=False):
    """
    optimize the constant coefficients in the candaidate expressions.

    candidate_ode_equations: list of strings for n ODE expressions. the discovered equation (with placeholders for coefficients).
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
    non_terminal_nodes: list of non terminal nodes. It is used for checking if the expression is valid

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

        print(candidate_ode_equations)
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

        def f(consts: list):
            # sttime = time.time()
            temp_equations = "$$".join(candidate_ode_equations)
            for i in range(len(consts)):
                temp_equations = temp_equations.replace('c' + str(i), str(consts[i]), 1)
            eq_est = temp_equations.split("$$")
            # usedtime = time.time() - sttime
            # print('step1', usedtime)
            # sys.stdout.flush()
            # sttime = time.time()
            pred_trajectories = execute(eq_est, init_cond, time_span, t_eval, input_var_Xs)
            # usedtime = time.time() - sttime
            # print('step2', usedtime)
            # sys.stdout.flush()
            # sttime = time.time()
            var_ytrue = np.var(true_trajectories)
            loss_val = -loss_func(pred_trajectories, true_trajectories, var_ytrue)
            # usedtime = time.time() - sttime
            # print('step3', usedtime)
            # sys.stdout.flush()
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
    print('\t loss:', train_loss,
          'eq:', candidate_ode_equations)

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
            one_solution = runge_kutta4(func, t_evals, one_x_init)
            pred_trajectories.append(one_solution)
        pred_trajectories = np.asarray(pred_trajectories)

        if pred_trajectories is complex:
            return None
            # return np.ones(init_cond.shape[-1]) * np.infty
    except TypeError as e:
        # print(e, expr, input_var_Xs, data_X.shape)
        # pred_trajectories = np.ones(init_cond.shape[-1]) * np.infty
        return None
    except KeyError as e:
        # print(e, expr)
        # pred_trajectories = np.ones(init_cond.shape[-1]) * np.infty
        return None
    except ValueError as e:
        return None
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


def sympy_plus_scipy():
    from sympy import symbols, lambdify
    import numpy as np
    import scipy.integrate

    # Create symbols y0, y1, and y2
    y = symbols('y:3')

    rf = y[0] ** 2 * y[1]
    rb = y[2] ** 2
    # Derivative of the function y(t); values for the three chemical species
    # for input values y, kf, and kb
    ydot = [2 * (rb - rf), rb - rf, 2 * (rf - rb)]
    print(ydot)
    t = symbols('t')  # not used in this case
    # Convert the SymPy symbolic expression for ydot into a form that
    # SciPy can evaluate numerically, f
    f = lambdify((t, y), ydot)
    k_vals = np.array([0.42, 0.17])  # arbitrary in this case
    y0 = [1, 0, 1]  # initial condition (initial values)
    y0 = np.asarray(y0)
    y0 = y0.T
    print(y0.shape)
    t_eval = np.linspace(0, 10, 50)  # evaluate integral from t = 0-10 for 50 points
    # Call SciPy's ODE initial value problem solver solve_ivp by passing it
    #   the function f,
    #   the interval of integration,
    #   the initial state, and
    #   the arguments to pass to the function f
    solution = scipy.integrate.solve_ivp(f, (0, 10), y0, t_eval=t_eval, vectorized=True)
    # Extract the y (concentration) values from SciPy solution result
    y = solution.y
    print(y.shape)
    # Plot the result graphically using matplotlib