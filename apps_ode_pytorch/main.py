import torch
import time
import click
import numpy as np
from scibench.symbolic_equation_evaluator import Equation_evaluator
from scibench.symbolic_data_generator import DataX

from grammar.grammar import ContextFreeGrammar
from grammar.grammar_regress_task import RegressTask
from grammar.production_rules import get_production_rules, construct_non_terminal_nodes_and_start_symbols
from grammar.grammar_program import grammarProgram
from active_deep_symbolic_regression import ActDeepSymbolicRegression

threshold_values = {
    'neg_mse': {'reward_threshold': -1e-6},
    'neg_nmse': {'reward_threshold': -1e-6},
    'neg_nrmse': {'reward_threshold': -1e-3},
    'neg_rmse': {'reward_threshold': -1e-3},
    'inv_mse': {'reward_threshold': 1 / (1 + 1e-6)},
    'inv_nmse': {'reward_threshold': 1 / (1 + 1e-6)},
    'inv_nrmse': {'reward_threshold': 1 / (1 + 1e-6)},
}


@click.command()
@click.argument('config_template', default="")
@click.option('--equation_name', default=None, type=str, help="Name of equation")
@click.option('--optimizer', default='BFGS', type=str, help="optimizer for the expressions")
@click.option('--metric_name', default='inv_nrmse', type=str, help="evaluation metrics")
@click.option('--num_init_conds', default=10, type=int, help="batch of initial condition of dataset")
@click.option('--num_regions', default=10, type=int, help="number of regions to be sampled")
@click.option('--noise_type', default='normal', type=str, help="")
@click.option('--noise_scale', default=0.0, type=float, help="")
@click.option('--max_len', default=10, help="max length of the sequence from the decoder")
@click.option('--total_iterations', default=100, help="Number of learning iterations")
@click.option('--n_cores', default=1, help="Number of cores for parallel evaluation")
@click.option('--use_gpu', default=-1, help="use GPU or cpu for training")
@click.option('--active_mode', default='default', help="use which active learning algorithm")
@click.option('--time_sequence_drop_rate', default=0, type=float, help="simulate irregular time sequence")
def main(config_template, optimizer, equation_name, metric_name, num_init_conds, num_regions, noise_type, noise_scale,
         max_len,
         total_iterations, n_cores, use_gpu, active_mode, time_sequence_drop_rate):
    data_query_oracle = Equation_evaluator(equation_name,
                                           noise_type, noise_scale,
                                           metric_name=metric_name)
    # print(data_query_oracle.vars_range_and_types_to_json)
    dataXgen = DataX(data_query_oracle.vars_range_and_types_to_json)
    nvars = data_query_oracle.get_nvars()
    function_set = data_query_oracle.get_operators_set()

    time_span = (0.0001, 2)
    trajectory_time_steps = 100
    max_opt_iter = 100
    t_eval = np.linspace(time_span[0], time_span[1], trajectory_time_steps)
    task = RegressTask(num_init_conds,
                       nvars,
                       dataXgen,
                       data_query_oracle,
                       time_span, t_eval,
                       num_of_regions=num_regions,
                       width=0.1)

    # get basic production rules
    reward_thresh = threshold_values[metric_name]
    non_terminal_nodes, start_symbols = construct_non_terminal_nodes_and_start_symbols(nvars)
    production_rules = []
    for one_nt_node in non_terminal_nodes:
        print(get_production_rules(nvars, function_set, one_nt_node))
        production_rules.extend(get_production_rules(nvars, function_set, one_nt_node))

    print("grammars:", production_rules)
    print("start_symbols:", start_symbols)
    program = grammarProgram(
        non_terminal_nodes=non_terminal_nodes,
        optimizer=optimizer,
        metric_name=metric_name,
        n_cores=n_cores,
        max_opt_iter=max_opt_iter
    )
    grammar_model = ContextFreeGrammar(
        nvars=nvars,
        production_rules=production_rules,
        start_symbols=start_symbols,
        non_terminal_nodes=non_terminal_nodes,
        max_length=max_len,
        topK_size=10,
        reward_threhold=reward_thresh
    )

    grammar_model.task = task
    grammar_model.program = program

    """Trains and returns dict of reward, expressions"""
    model = ActDeepSymbolicRegression(config_template, grammar_model)

    # Establish GPU device if necessary
    if use_gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(use_gpu))
    else:
        device = torch.device("cpu")

    start = time.time()
    print("deep model setup.....")

    model.setup(device)
    epoch_best_rewards, epoch_best_expressions, best_reward, best_expression = model.train(
        threshold_values[metric_name]['reward_threshold'],
        total_iterations,
        active_mode
    )

    # Plot best rewards each epoch
    print(epoch_best_rewards)
    print(epoch_best_expressions)
    print(best_reward)
    print(best_expression)

    #####
    end_time = time.time() - start

    grammar_model.print_topk_expressions(verbose=True)
    print("APPS time {} mins".format(np.round(end_time / 60, 3)))


if __name__ == '__main__':
    main()
