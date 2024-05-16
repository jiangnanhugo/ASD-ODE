import time
import argparse
import os
import sys

from mcts_model import MCTS
from production_rules import  get_production_rules
from sympy import lambdify, Symbol

import random
import numpy as np
from scibench.symbolic_data_generator import DataX, compute_time_derivative
from scibench.symbolic_equation_evaluator import Equation_evaluator
from grammar.grammar_regress_task import RegressTask
from program import Program


def run_mcts(
        production_rules, non_terminal_nodes=['A'], num_episodes=1000, num_rollouts=40,
        max_len=30, eta=0.9999, max_module_init=15, num_aug=10, exp_rate=1 / np.sqrt(2),
        num_transplant=1, norm_threshold=1e-10
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

        start = time.time()
        _, good_modules = mcts_model.MCTS_run_orig(num_episodes,
                                                   num_rollouts=num_rollouts,
                                                   verbose=True,
                                                   is_first_round=True,
                                                   print_freq=5)

        mcts_model.print_hofs(-2, verbose=True)

        if not best_modules:
            best_modules = good_modules
        else:
            best_modules = sorted(list(set(best_modules + good_modules)), key=lambda x: x[1])

        aug_grammars = [x[0] for x in best_modules[:num_aug]]
        print("AUG Grammars")
        for gi in aug_grammars:
            print(gi)

        max_module += module_grow_step
        exploration_rate *= 1.2
    print("final hof")
    mcts_model.print_hofs(-2, verbose=True)


def mcts(equation_name, num_init_conds, metric_name, noise_type, noise_scale, optimizer, num_episodes):
    data_query_oracle = Equation_evaluator(equation_name, num_init_conds,
                                           noise_type, noise_scale,
                                           metric_name=metric_name)
    dataX = DataX(data_query_oracle.vars_range_and_types_to_json)
    nvars = data_query_oracle.get_nvars()
    operators_set=data_query_oracle.operators_set
    input_var_Xs = [Symbol(f'X{i}') for i in range(nvars)]
    time_span = (0.0001, 10)
    trajectory_time_steps = 100

    t_eval = np.linspace(time_span[0], time_span[1], trajectory_time_steps)
    task = RegressTask(num_init_conds,
                       nvars,
                       dataX,
                       data_query_oracle,
                       time_span, t_eval)

    task.rand_draw_init_cond()
    true_trajectories = task.evaluate()
    X_train, y_train = compute_time_derivative(true_trajectories, t_eval)

    print("X_train.shape: {}, y_train.shape: {}".format(X_train.shape, y_train.shape))
    sys.stdout.flush()
    # data_query_oracle = Equation_evaluator(equation_name, noise_type, noise_scale, metric_name)
    # dataXgen = DataX(data_query_oracle.get_vars_range_and_types())
    # nvar = data_query_oracle.get_nvars()
    # operators_set = data_query_oracle.get_operators_set()
    #
    # regress_batchsize = 256
    allowed_input_tokens = np.ones(nvars, dtype=np.int32)
    MCTS.task = task
    MCTS.program = Program(nvars, optimizer)
    MCTS.program.evalaute_loss = data_query_oracle.compute_metric

    production_rules = get_production_rules(nvars, operators_set)
    print("The production rules are:", production_rules)
    start = time.time()
    run_mcts(production_rules=production_rules, num_episodes=num_episodes)
    end_time = time.time() - start
    print("MCTS {} mins".format(np.round(end_time / 60, 3)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--equation_name", help="the filename of the true program.")
    parser.add_argument("--metric_name", type=str, default='neg_mse', help="The name of the metric for loss.")

    parser.add_argument("--num_init_conds", type=int, default=5, help="batch of initial condition of dataset.")
    parser.add_argument('--optimizer',
                        nargs='?',
                        choices=['BFGS', 'L-BFGS-B', 'Nelder-Mead', 'CG', 'basinhopping', 'dual_annealing', 'shgo',
                                 'direct'],
                        help='list servers, storage, or both (default: %(default)s)')

    parser.add_argument("--num_episodes", type=int, default=1000, help="the number of episode for MCTS.")
    parser.add_argument("--num_per_episodes", type=int, default=30, help="the number of episode for MCTS.")
    parser.add_argument("--noise_type", type=str, default='normal', help="The name of the noises.")
    parser.add_argument("--noise_scale", type=float, default=0.0,
                        help="This parameter adds the standard deviation of the noise")

    args = parser.parse_args()

    seed = int(time.perf_counter() * 10000) % 1000007
    random.seed(seed)
    print('random seed=', seed)

    seed = int(time.perf_counter() * 10000) % 1000007
    np.random.seed(seed)
    print('np.random seed=', seed)
    print(args)

    # run Monte Carlo Tree Search
    mcts(args.equation_name, args.metric_name, args.num_init_conds, args.noise_type, args.noise_scale, args.optimizer,
         args.num_episodes
         )
