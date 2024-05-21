import time
import argparse
import os
import sys

from mcts_model import MCTS
from production_rules import get_production_rules
from sympy import lambdify, Symbol

import random
import numpy as np
from scibench.symbolic_data_generator import DataX, compute_time_derivative
from scibench.symbolic_equation_evaluator import Equation_evaluator
from grammar.grammar_regress_task import RegressTask
from program import Program
from regress_task import symbolicRegressionTask
from grammar.minimize_coefficients import execute


class SymbolicDifferentialEquations(object):
    """
    this class is used to represent symbolic ordinary differential equations.
    For n variables settings, there will be n total expressions.
    """

    def __init__(self, list_of_sympy_equations):
        # self.traversal = list_of_rules
        self.fitted_eq = list_of_sympy_equations
        self.all_metrics = None

    def __repr__(self):
        return "Eq=[{}]".format(",\t ".join(self.fitted_eq))


def print_expressions(pr, task, input_var_Xs):
    pred_trajectories = execute(pr.fitted_eq,
                                task.init_cond, task.time_span, task.t_evals,
                                input_var_Xs)
    dict_of_result = task.evaluate_all_losses(pred_trajectories)
    print('-' * 30)
    print(pr.fitted_eq)
    for metric_name in dict_of_result:
        print(f"{metric_name} {dict_of_result[metric_name]}")
    print('-' * 30)


def run_mcts(
        production_rules, non_terminal_nodes=['A'], num_episodes=1000, num_rollouts=20,
        max_len=30, eta=0.9999, max_module_init=15, num_aug=10, exp_rate=1 / np.sqrt(2),
        num_transplant=1,
):
    """
    production_rules: rules to generate expressions
    num_episodes: number of iterations.
    non_terminal_nodes: used in production rules
    num_rollouts
    max_len: maximum allowed length (number of production rules ) of discovered equations.
    eta: penalty factor for rewarding.
    max_module_init:  initial maximum length for module transplantation candidates.
    num_aug : number of trees for module transplantation.
    exp_rate: initial exploration rate.
    norm_threshold: numerical error tolerance for norm calculation, a very small value.
    """

    # define production rules and non-terminal nodes.
    grammars = production_rules

    # number of module max size increase after each transplantation
    module_grow_step = (max_len - max_module_init) / num_transplant

    exploration_rate = exp_rate
    max_module = max_module_init
    best_modules = []
    aug_grammars = []

    for i_itr in range(num_transplant):
        print("transplanation step=", i_itr)
        print("aug_grammars:", aug_grammars)
        max_opt_iter = 200

        mcts_model = MCTS(base_grammars=grammars,
                          aug_grammars=aug_grammars,
                          non_terminal_nodes=non_terminal_nodes,
                          aug_nt_nodes=[],
                          max_len=max_len,
                          max_module=max_module,
                          aug_grammars_allowed=num_aug,
                          exploration_rate=exploration_rate,
                          max_opt_iter=max_opt_iter,
                          eta=eta)

        _, hall_of_fame = mcts_model.MCTS_run_orig(
            num_episodes,
            num_rollouts=num_rollouts,
            verbose=True,
            print_freq=5)

        mcts_model.print_hofs(verbose=True)

        if not best_modules:
            best_modules = hall_of_fame
        else:
            best_modules = sorted(list(set(best_modules + hall_of_fame)), key=lambda x: x[1])

        aug_grammars = [x[0] for x in best_modules[:num_aug]]
        print("AUG Grammars")
        for gi in aug_grammars:
            print(gi)

        max_module += module_grow_step
        exploration_rate *= 1.2
    print("final hof")
    mcts_model.print_hofs(verbose=True)
    return hall_of_fame[-1][-1]


def mcts(equation_name, num_init_conds, metric_name, noise_type, noise_scale, num_episodes,
         time_sequence_drop_rate=0.0):
    optimizer = 'BFGS'

    data_query_oracle = Equation_evaluator(equation_name,
                                           noise_type, noise_scale,
                                           metric_name=metric_name,
                                           time_sequence_drop_rate=time_sequence_drop_rate)
    dataX = DataX(data_query_oracle.vars_range_and_types_to_json)
    nvars = data_query_oracle.get_nvars()
    operators_set = data_query_oracle.operators_set
    input_var_Xs = [Symbol(f'X{i}') for i in range(nvars)]
    time_span = (0.0001, 10)
    trajectory_time_steps = 1000

    t_eval = np.linspace(time_span[0], time_span[1], trajectory_time_steps)
    task = RegressTask(num_init_conds,
                       nvars,
                       dataX,
                       data_query_oracle,
                       time_span, t_eval)

    task.rand_draw_init_cond()
    true_trajectories = task.evaluate()
    random_mask = np.random.choice([0, 1],
                                   size=(true_trajectories.shape[0], true_trajectories.shape[1]),
                                   p=[time_sequence_drop_rate, 1 - time_sequence_drop_rate])

    # Expand the mask to match the shape of traj and pred_traj
    expanded_mask = np.expand_dims(random_mask, axis=2)
    true_trajectories = true_trajectories * expanded_mask
    X_train, y_train = compute_time_derivative(true_trajectories, t_eval)

    print("X_train.shape: {}, y_train.shape: {}".format(X_train.shape, y_train.shape))
    sys.stdout.flush()

    one_predict_ode = []
    for xi in range(nvars):
        MCTS.task = symbolicRegressionTask(
            batchsize=100,
            n_input=nvars,
            X_train=X_train[:, :, xi].reshape(-1, 1),
            y_train=y_train[:, :, xi].reshape(-1, 1),
            metric_name=metric_name,
        )
        MCTS.program = Program(nvars, optimizer)
        MCTS.program.evalaute_loss = MCTS.task.metric

        production_rules = get_production_rules(nvars, operators_set)
        print("The production rules are:", production_rules)
        start = time.time()
        model_str = run_mcts(production_rules=production_rules, num_episodes=num_episodes)
        end_time = time.time() - start
        print("SPL {} mins".format(np.round(end_time / 60, 3)))
        one_predict_ode.append(str(model_str))
    temp = SymbolicDifferentialEquations(one_predict_ode)
    print_expressions(temp, task, input_var_Xs)
    print("=" * 20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--equation_name", help="the filename of the true program.")
    parser.add_argument("--metric_name", type=str, default='neg_mse', help="The name of the metric for loss.")

    parser.add_argument("--num_init_conds", type=int, default=50, help="batch of initial condition of dataset.")

    parser.add_argument("--num_episodes", type=int, default=10, help="the number of episode for MCTS.")
    parser.add_argument("--noise_type", type=str, default='normal', help="The name of the noises.")
    parser.add_argument("--noise_scale", type=float, default=0.0,
                        help="This parameter adds the standard deviation of the noise")
    parser.add_argument("--time_sequence_drop_rate", type=float, default=0.0,
                        help="simulate irregular time sequence")

    args = parser.parse_args()

    seed = int(time.perf_counter() * 10000) % 1000007
    random.seed(seed)
    print('random seed=', seed)

    seed = int(time.perf_counter() * 10000) % 1000007
    np.random.seed(seed)
    print('np.random seed=', seed)
    print(args)

    # run Monte Carlo Tree Search
    mcts(args.equation_name, args.num_init_conds, args.metric_name, args.noise_type, args.noise_scale,
         args.num_episodes, args.time_sequence_drop_rate)
