# given a list of expressions, predict the correct ranking of them using differnt active learning strategies.
# strategies: 1. phase_portait 2. full data 3. QBC (query by committee) with different entropy objective
# 4, qusor
import copy

import time
import click
import numpy as np
from scibench.symbolic_equation_evaluator import Equation_evaluator
from scibench.symbolic_data_generator import DataX
from sympy.parsing import parse_expr
from grammar.grammar import ContextFreeGrammar
from grammar.grammar_regress_task import RegressTask
from grammar.grammar_program import grammarProgram
from grammar.minimize_coefficients import execute


class SymbolicDifferentialEquations(object):
    """
    this class is used to represent symbolic ordinary differential equations.
    For n variables settings, there will be n total expressions.
    """

    def __init__(self, one_sympy_equations):
        # self.traversal = list_of_rules
        self.fitted_eq = one_sympy_equations
        self.all_metrics = None

    def __repr__(self):
        return str(self.fitted_eq)


def print_expressions(pr, task, input_var_Xs):
    pred_trajectories = execute(pr.fitted_eq,
                                task.init_cond, task.time_span, task.t_evals,
                                input_var_Xs)
    dict_of_result = task.evaluate_all_losses(pred_trajectories)
    print('-' * 30)
    for metric_name in dict_of_result:
        print(f"{metric_name} {dict_of_result[metric_name]}")
    print('-' * 30)


def read_expressions_from_file(expression_file):
    content = open(expression_file, 'r').readlines()
    expressions = []
    for line in content:
        spl = [one_eq.strip() for one_eq in line.strip()[1:-1].split(",")]
        expressions.append(SymbolicDifferentialEquations(spl))
    return expressions


def normalised_kendall_tau_distance(values1, values2):
    """Compute the Kendall tau distance.
    https://en.wikipedia.org/wiki/Kendall_tau_distance
    """
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(values1)
    b = np.argsort(values2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]),
                                np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered / (n * (n - 1))


@click.command()
@click.option('--equation_name', default="vars2_prog10", type=str, help="Name of equation")
@click.option('--pred_expressions_file', default='/home/jiangnan/PycharmProjects/act_ode/ablation_study/pred_expressions/vars2_prog10.out', type=str,
              help="optimizer for the expressions")
@click.option('--num_init_conds', default=10, type=int, help="batch of initial condition of dataset")
@click.option('--num_regions', default=10, type=int, help="number of regions to be sampled")
@click.option('--region_width', default=0.1, type=float, help="")
@click.option('--noise_type', default='normal', type=str, help="")
@click.option('--noise_scale', default=0.0, type=float, help="")
@click.option('--active_mode', default='phase_portrait', help="Number of cores for parallel evaluation")
@click.option('--full_mesh_size', default=1000, type=int, help="")
def main(equation_name, pred_expressions_file, num_init_conds, num_regions, region_width, noise_type, noise_scale,
         active_mode, full_mesh_size):
    data_query_oracle = Equation_evaluator(equation_name, num_init_conds,   noise_type, noise_scale, metric_name='neg_mse')
    grammar_expressions = read_expressions_from_file(pred_expressions_file)
    dataXgen = DataX(data_query_oracle.vars_range_and_types_to_json)
    nvars = data_query_oracle.get_nvars()

    time_span = (0.0001, 2)
    trajectory_time_steps = 1000
    t_eval = np.linspace(time_span[0], time_span[1], trajectory_time_steps)
    task = RegressTask(num_init_conds,
                       nvars,
                       dataXgen,
                       data_query_oracle,
                       time_span, t_eval,
                       num_of_regions=num_regions,
                       width=region_width)

    # get basic production rules
    program = grammarProgram(
        input_var_Xs='X1',
        optimizer='BFGS',
        metric_name='neg_mse',
        n_cores=1,
        max_opt_iter=0
    )
    grammar_model = ContextFreeGrammar(
        nvars=nvars,
        production_rules=[],
        start_symbols='A',
        non_terminal_nodes='X1',
        max_length=20,
        topK_size=len(grammar_expressions),
        reward_threhold=0
    )

    grammar_model.task = task
    grammar_model.program = program

    print("deep model setup.....")
    temp = copy.deepcopy(grammar_expressions)
    start = time.time()
    correct_topk = grammar_model.expression_active_evalution(temp,
                                                             active_mode='full',
                                                             full_mesh_size=full_mesh_size)
    end_time = time.time() - start

    print("{} time {} mins".format(active_mode, np.round(end_time / 60, 3)))
    #####
    temp = copy.deepcopy(grammar_expressions)
    start = time.time()
    grammar_expressions = grammar_model.expression_active_evalution(temp, active_mode=active_mode)
    #####
    end_time = time.time() - start
    # Update the best set of expressions discovered
    for p in grammar_expressions:
        grammar_model.update_topK_expressions(p)
    topk = grammar_expressions

    grammar_model.print_topk_expressions(verbose=True)
    print("{} time {} mins".format(active_mode, np.round(end_time / 60, 3)))
    kendall_tau_score = normalised_kendall_tau_distance(topk, correct_topk)
    print("kendall_tau_score: {}".format(kendall_tau_score))


if __name__ == '__main__':
    main()
