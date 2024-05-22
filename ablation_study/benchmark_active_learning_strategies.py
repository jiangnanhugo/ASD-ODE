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
from grammar.act_sampling import deep_coreset
from scipy.stats import kendalltau
from itertools import combinations


class SymbolicDifferentialEquations(object):
    """
    this class is used to represent symbolic ordinary differential equations.
    For n variables settings, there will be n total expressions.
    """

    def __init__(self, one_sympy_equations):
        self.fitted_eq = one_sympy_equations
        self.train_loss = 0.0
        self.all_metrics = None
        self.valid_loss = None

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


def normalised_kendall_tau_distance(array1, array2):
    """Compute the Kendall tau distance.
    https://en.wikipedia.org/wiki/Kendall_tau_distance
    """
    # Extract valid_loss values
    valid_losses1 = np.array([obj.valid_loss for obj in array1])[:3]

    valid_losses2 = np.array([obj.valid_loss for obj in array2])[:3]

    # Compute the Kendall Tau correlation coefficient
    tau, _ = kendalltau(valid_losses1, valid_losses2)

    # Compute the Kendall Tau distance
    def kendall_tau_distance(tau, n):
        return (1 - tau) * n * (n - 1) / 2

    n = len(valid_losses1)  # Length of the rankings
    distance = kendall_tau_distance(tau, n)

    # Normalize the Kendall Tau distance
    max_distance = n * (n - 1) / 2
    normalized_distance = distance / max_distance

    # print(f"Normalized Kendall Tau Distance: {normalized_distance}")
    return normalized_distance


def enlarge_effect(array1, array2):
    array1 = np.array([obj.valid_loss for obj in array1])

    array2 = np.array([obj.valid_loss for obj in array2])
    cnt0, cnt_p1, cnt_n1 = 0, 0, 0
    for i in range(len(array1)):
        for j in range(i + 1, len(array1)):
            if abs(array1[i] - array1[j]) <= abs(array2[i] - array2[j]):
                cnt_p1 += 1
            else:
                cnt_n1 += 1
    return cnt_p1, cnt_n1


@click.command()
@click.option('--equation_name', default="vars2_prog10", type=str, help="Name of equation")
@click.option('--pred_expressions_file',
              default='/home/jiangnan/PycharmProjects/act_ode/ablation_study/pred_expressions/vars2_prog10.out',
              type=str,
              help="optimizer for the expressions")
@click.option('--num_init_conds', default=20, type=int, help="batch of initial condition of dataset")
@click.option('--num_regions', default=20, type=int, help="number of regions to be sampled")
@click.option('--region_width', default=0.1, type=float, help="")
@click.option('--noise_type', default='normal', type=str, help="")
@click.option('--noise_scale', default=0.0, type=float, help="")
@click.option('--active_mode', default='phase_portrait', help="Number of cores for parallel evaluation")
@click.option('--full_mesh_size', default=100, type=int, help="")
def main(equation_name, pred_expressions_file, num_init_conds, num_regions, region_width, noise_type, noise_scale,
         active_mode, full_mesh_size):
    data_query_oracle = Equation_evaluator(equation_name, noise_type, noise_scale,
                                           metric_name='neg_mse')
    grammar_expressions = read_expressions_from_file(pred_expressions_file)
    dataXgen = DataX(data_query_oracle.vars_range_and_types_to_json)
    nvars = data_query_oracle.get_nvars()

    time_span = (0.01, 0.1)
    trajectory_time_steps = 10
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
        non_terminal_nodes='X1',
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
    num_repetitions = 5

    phase_pred_list = []
    default_pred_list = []
    region = None
    for i in range(num_repetitions):
        #####
        temp = copy.deepcopy(grammar_expressions)
        start = time.time()
        top_pred_default = grammar_model.expression_active_evaluation(temp, active_mode='default')
        default_pred_list.append(top_pred_default)
        #####
        end_time = time.time() - start
        # Update the best set of expressions discovered

        print("default time {} mins".format(np.round(end_time / 60, 3)))

        #####
        temp = copy.deepcopy(grammar_expressions)
        start = time.time()
        task.init_cond = task.full_init_cond(full_mesh_size)
        true_traj=task.evaluate()
        true_traj = true_traj.reshape(true_traj.shape[0], -1)
        deep_coreset(true_traj)
        # top_pred_default = grammar_model.expression_active_evaluation(temp, active_mode='default')
        default_pred_list.append(top_pred_default)
        #####
        end_time = time.time() - start
        # Update the best set of expressions discovered

        print("default time {} mins".format(np.round(end_time / 60, 3)))
        #####

        start = time.time()
        temp = copy.deepcopy(grammar_expressions)
        top_pred = grammar_model.expression_active_evaluation(temp, active_mode=active_mode, given_region=region)
        region = grammar_model.regions
        phase_pred_list.append(top_pred)
        #####
        end_time = time.time() - start
        # Update the best set of expressions discovered

        print("{} time {} mins".format(active_mode, np.round(end_time / 60, 3)))

        ###
        pos, neg = enlarge_effect(top_pred_default, top_pred)
        print("pos={} neg={}".format(pos, neg))
        ###

    #
    temp = copy.deepcopy(grammar_expressions)
    start = time.time()
    # correct_topk = grammar_model.expression_active_evalution(temp,
    #                                                          active_mode='full',
    #                                                          full_mesh_size=full_mesh_size)
    end_time = time.time() - start

    print("full time {} mins".format(np.round(end_time / 60, 3)))
    #####
    total_score = 0
    for one, another in combinations(phase_pred_list, 2):
        kendall_tau_score = normalised_kendall_tau_distance(one, another)
        total_score += kendall_tau_score
        print("kendall_tau_score between {} and full data : {}".format(active_mode, kendall_tau_score))
        #
    total_score_default = 0.0
    for one, another in combinations(default_pred_list, 2):
        kendall_tau_score = normalised_kendall_tau_distance(one, another)
        total_score_default += kendall_tau_score
        print("kendall_tau_score between default and full data : {}".format(kendall_tau_score))
    print(f"toal score of phase portrait: {total_score}, default {total_score_default}")


if __name__ == '__main__':
    main()
