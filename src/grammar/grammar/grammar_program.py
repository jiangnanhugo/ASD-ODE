"""Class for symbolic expression optimization."""

from sympy.parsing.sympy_parser import parse_expr
from sympy import lambdify, symbols

from itertools import chain
import sys
import numpy as np
import warnings

from scipy.optimize import minimize
from scipy.optimize import basinhopping, shgo, dual_annealing, direct
from scipy.integrate import solve_ivp

from grammar.grammar_utils import pretty_print_expr
from grammar.production_rules import concate_production_rules_to_expr, check_non_terminal_nodes
from grammar.metrics import all_metrics

from pathos.multiprocessing import ProcessPool

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.set_printoptions(precision=4, linewidth=np.inf)


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
        return " reward={}, equation=\n{}".format(self.reward, "\n ".join(self.fitted_eq))

    def print_all_metrics(self):
        print('-' * 30)
        if not self.all_metrics:
            print("No metrics")
            print('-' * 30)
            return
        for metric_name in self.all_metrics:
            print(f"{metric_name} {self.all_metrics[metric_name]}")
        print('-' * 30)


class grammarProgram(object):
    """
    used for optimizing the constants in the expressions.
    """
    evaluate_loss = None

    def __init__(self, non_terminal_nodes, optimizer="BFGS", metric_name='inv_nrmse', max_opt_iter=100, n_cores=1,
                 max_open_constants=20):
        """
        max_open_constants: the maximum number of allowed open constants in the expression.
        """
        self.optimizer = optimizer
        self.max_opt_iter = max_opt_iter
        self.max_open_constants = max_open_constants
        self.metric_name = metric_name
        self.n_cores = n_cores
        self.non_terminal_nodes = non_terminal_nodes
        self.evaluate_loss = all_metrics[metric_name]
        if self.n_cores>1:
            self.pool = ProcessPool(nodes=self.n_cores)

    def fitting_new_expressions(self, many_seqs_of_rules,
                                init_cond: np.ndarray, time_span, t_eval,
                                true_trajectories,
                                input_var_Xs):
        """
        we assume the input must be a valid expression
        init_cond: [batch_size, nvars].
        true_trajectories: [batch_size, time_steps, nvars]. the correct trajectories.
        """
        result = []
        print("many_seqs_of_rules:", len(many_seqs_of_rules))

        for i, one_list_rules in enumerate(many_seqs_of_rules):
            one_expr = SymbolicDifferentialEquations(one_list_rules)
            reward, fitted_eq, _, _ = optimize(
                one_expr.expr_template,
                init_cond, time_span, t_eval,
                true_trajectories,
                input_var_Xs,
                self.evaluate_loss,
                self.max_open_constants,
                self.max_opt_iter,
                self.optimizer,
                self.non_terminal_nodes
            )

            one_expr.reward = reward
            one_expr.fitted_eq = fitted_eq
            result.append(one_expr)
            print('idx=', i)
            sys.stdout.flush()
        return result

    def fitting_new_expressions_in_parallel(self, many_seqs_of_rules, init_cond: np.ndarray, time_span, t_eval,
                                            true_trajectories,
                                            input_var_Xs):
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
        init_cond_ncores = [init_cond for _ in range(self.n_cores)]
        true_trajectories_ncores = [true_trajectories for _ in range(self.n_cores)]
        input_var_Xes = [input_var_Xs for _ in range(self.n_cores)]
        time_span_ncores = [time_span for _ in range(self.n_cores)]
        t_eval_ncores = [t_eval for _ in range(self.n_cores)]
        evaluate_losses = [self.evaluate_loss for _ in range(self.n_cores)]
        max_open_constantes = [self.max_open_constants for _ in range(self.n_cores)]
        max_opt_iteres = [self.max_opt_iter for _ in range(self.n_cores)]
        optimizeres = [self.optimizer for _ in range(self.n_cores)]
        non_terminal_nodes = [self.non_terminal_nodes for _ in range(self.n_cores)]

        result = self.pool.map(fit_one_expr, many_expr_tempaltes, init_cond_ncores, time_span_ncores, t_eval_ncores,
                               true_trajectories_ncores,
                               input_var_Xes, evaluate_losses,
                               max_open_constantes, max_opt_iteres, optimizeres, non_terminal_nodes)
        result = list(chain.from_iterable(result))
        print("Done with optimization!")
        sys.stdout.flush()
        return result


def fit_one_expr(one_expr_batch, init_cond, time_span, t_eval, true_trajectories, input_var_Xs, evaluate_loss,
                 max_open_constants, max_opt_iter,
                 optimizer_name, non_terminal_nodes):
    results = []
    for one_expr in one_expr_batch:
        reward, fitted_eq, _, _ = optimize(one_expr.expr_template,
                                           init_cond, time_span, t_eval,
                                           true_trajectories,
                                           input_var_Xs,
                                           evaluate_loss, max_open_constants, max_opt_iter, optimizer_name,
                                           non_terminal_nodes)

        one_expr.reward = reward
        one_expr.fitted_eq = fitted_eq
        results.append(one_expr)
    return results


def optimize(candidate_ode_equations: list, init_cond, time_span, t_eval, true_trajectories, input_var_Xs,
             evaluate_loss, max_open_constants, max_opt_iter,
             optimizer_name,
             non_terminal_nodes,
             user_scpeficied_iters=-1,
             verbose=False):
    """
    Calculate reward score for a complete parse tree
    If placeholder C is in the equation, also execute estimation for C
    Reward = 1 / (1 + MSE) * Penalty ** num_term_in_expressions

    candidate_ode_equations : list of expressions. the discovered equation (with placeholders for coefficients).
    init_cond: [batch_size, nvars]. the initial conditions of each variables.
    true_trajectories: [batch_size, time_steps, nvars]. the true trajectories.
    """

    candidate_ode_equations = simplify_template(candidate_ode_equations)
    print("candidate:", candidate_ode_equations)
    if check_non_terminal_nodes(candidate_ode_equations, non_terminal_nodes):  # not a valid equation
        return -np.inf, candidate_ode_equations, 0, np.inf

    # count the total number of constants in equation
    num_changing_consts = sum([x.count('C') for x in candidate_ode_equations])
    t_optimized_constants, t_optimized_obj = 0, np.inf
    if num_changing_consts == 0:  # zero constant
        var_ytrue = np.var(true_trajectories)
        pred_trajectories = execute(candidate_ode_equations, init_cond, time_span, t_eval, input_var_Xs)
    elif num_changing_consts >= max_open_constants:  # discourage over complicated numerical estimations
        return -np.inf, candidate_ode_equations, t_optimized_constants, t_optimized_obj
    else:
        c_lst = ['c' + str(i) for i in range(num_changing_consts)]
        temp_equations = "$$".join(candidate_ode_equations)
        for c in c_lst:
            temp_equations = temp_equations.replace('C', c, 1)
        candidate_ode_equations = temp_equations.split("$$")

        def f(consts: list):
            temp_equations = "$$".join(candidate_ode_equations)
            for i in range(len(consts)):
                temp_equations = temp_equations.replace('c' + str(i), str(consts[i]), 1)
            eq_est = temp_equations.split("$$")

            pred_trajectories = execute(eq_est, init_cond, time_span, t_eval, input_var_Xs)
            var_ytrue = np.var(true_trajectories)
            loss_val = -evaluate_loss(pred_trajectories, true_trajectories, var_ytrue)
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


    # r = eta ** tree_size * float(-np.log10(1e-60 - self.evaluate_loss(pred_trajectories, y_true, var_ytrue)))
    print('\t loss:', evaluate_loss(pred_trajectories, true_trajectories, var_ytrue),
          'eq:', candidate_ode_equations)
    reward = evaluate_loss(pred_trajectories, true_trajectories, var_ytrue)

    return reward, candidate_ode_equations, t_optimized_constants, t_optimized_obj


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


def execute(expr_strs: list, x_init_conds: np.ndarray, time_span: tuple, t_eval: np.ndarray,
            input_var_Xs: list) -> np.ndarray:
    """
    given a symbolic ODE (func) and the initial condition (init_cond), compute the time trajectory.
    https://docs.sympy.org/latest/guides/solving/solve-ode.html
    expr_strs: list of string. each string is one expression.
    time_span: tuple
    t_evals: np.linspace, or np.logspace
    x_init_conds: [batch_size, nvars]
    pred_trajectories: [batch_size, time_steps, nvars]
    """

    expr_odes = [parse_expr(one_expr) for one_expr in expr_strs]
    t = symbols('t')  # not used in this case
    try:
        func = lambdify((t, input_var_Xs), expr_odes, 'numpy')
        pred_trajectories = []
        for one_x_init in x_init_conds:
            one_solution = solve_ivp(func, t_span=time_span, y0=one_x_init, t_eval=t_eval)
            pred_trajectories.append(one_solution.y)
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


if __name__ == '__main__':
    expr_temp = 'sqrt(sqrt(C))*(sqrt(X0)+C)'
    simplify_template([expr_temp])


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
