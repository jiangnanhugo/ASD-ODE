"""Class for symbolic expression optimization."""
from itertools import chain
import sys
import numpy as np
import warnings

from grammar.production_rules import concate_production_rules_to_expr
from grammar.metrics import all_metrics

from pathos.multiprocessing import ProcessPool
from grammar.minimize_coefficients import optimize

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
        self.valid_loss = None
        self.train_loss = None
        self.fitted_eq = None
        self.invalid = False
        self.all_metrics = None

    def __repr__(self):
        return " train_loss={:.14f}\t valid_loss={:.14f}\t Eq=[{}]".format(
            self.train_loss, self.valid_loss, ",\t ".join(self.fitted_eq))

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

    def __init__(self, non_terminal_nodes, optimizer="BFGS", metric_name='neg_mse', max_opt_iter=1000, n_cores=1,
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
        self.loss_func = all_metrics[metric_name]
        if self.n_cores > 1:
            self.pool = ProcessPool(nodes=self.n_cores)

    def fitting_new_expressions(self, many_seqs_of_rules,
                                init_cond: np.ndarray, time_span, t_eval,
                                true_trajectories,
                                input_var_Xs):
        """
        fit the coefficients in the candidate ODEs.
        init_cond: [batch_size, nvars].
        true_trajectories: [batch_size, time_steps, nvars]. the correct trajectories.
        """
        result = []
        print("many_seqs_of_rules:", len(many_seqs_of_rules))

        for i, one_list_rules in enumerate(many_seqs_of_rules):
            one_expr = SymbolicDifferentialEquations(one_list_rules)
            train_loss, fitted_eq, _, _ = optimize(
                one_expr.expr_template,
                init_cond, time_span, t_eval,
                true_trajectories,
                input_var_Xs,
                self.loss_func,
                self.max_open_constants,
                self.max_opt_iter,
                self.optimizer,
                self.non_terminal_nodes
            )

            one_expr.train_loss = train_loss
            one_expr.fitted_eq = fitted_eq
            result.append(one_expr)
            print('idx=', i,f"/{len(many_seqs_of_rules)}")

            sys.stdout.flush()
        return result

    def fitting_new_expressions_in_parallel(self, many_seqs_of_rules, init_cond: np.ndarray, time_span, t_eval,
                                            true_trajectories,
                                            input_var_Xs):
        """
        fit the coefficients in many ODE in parallel
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
        evaluate_losses = [self.loss_func for _ in range(self.n_cores)]
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


def fit_one_expr(one_expr_batch, init_cond, time_span, t_eval, true_trajectories, input_var_Xs, loss_func,
                 max_open_constants, max_opt_iter,
                 optimizer_name, non_terminal_nodes):
    results = []
    for one_expr in one_expr_batch:
        train_loss, fitted_eq, _, _ = optimize(
            one_expr.expr_template,
            init_cond, time_span, t_eval,
            true_trajectories,
            input_var_Xs,
            loss_func, max_open_constants, max_opt_iter, optimizer_name,
            non_terminal_nodes)

        one_expr.train_loss = train_loss
        one_expr.fitted_eq = fitted_eq
        results.append(one_expr)

    return results

