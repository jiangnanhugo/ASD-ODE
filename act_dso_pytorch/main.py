# main.py: From here, launch deep symbolic regression tasks. All
# hyperparameters are exposed (info on them can be found in train.py). Unless
# you'd like to impose new constraints / make significant modifications,
# modifying this file (and specifically the get_data function) is likely all
# you need to do for a new symbolic regression task.


from train import learn
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
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
from utils import load_config

#
from expression_decoder import NeuralExpressionDecoder


threshold_values = {
    'neg_mse': {'reward_threshold': 1e-6},
    'neg_nmse': {'reward_threshold': 1e-6},
    'neg_nrmse': {'reward_threshold': 1e-3},
    'neg_rmse': {'reward_threshold': 1e-3},
    'inv_mse': {'reward_threshold': 1 / (1 + 1e-6)},
    'inv_nmse': {'reward_threshold': 1 / (1 + 1e-6)},
    'inv_nrmse': {'reward_threshold': 1 / (1 + 1e-6)},
}


###############################################################################
# Main Function
###############################################################################

# A note on operators:
# Available operators are: '*', '+', '-', '/', '^', 'sin', 'cos', 'tan',
#   'sqrt', 'square', and 'c.' You may also include constant floats, but they
#   must be strings. For variable operators, you must use the prefix var_.
#   Variable should be passed in the order that they appear in your data, i.e.
#   if your input data is structued [[x1, y1] ... [[xn, yn]] with outputs
#   [z1 ... zn], then var_x should precede var_y.

@click.command()
@click.argument('config_template', default="")
@click.option('--equation_name', default=None, type=str, help="Name of equation")
@click.option('--optimizer', default='BFGS', type=str, help="optimizer for the expressions")
@click.option('--metric_name', default='inv_nrmse', type=str, help="evaluation metrics")
@click.option('--num_init_conds', default=10, type=int, help="batch of initial condition of dataset")
@click.option('--noise_type', default='normal', type=str, help="")
@click.option('--noise_scale', default=0.0, type=float, help="")
@click.option('--max_len', default=10, help="max length of the sequence from the decoder")
@click.option('--total_iterations', default=20, help="Number of iterations per rounds")
@click.option('--n_cores', default=1, help="Number of cores for parallel evaluation")
@click.option('--use_gpu', default=-1, help="Number of cores for parallel evaluation")
def main(config_template, optimizer, equation_name, metric_name, num_init_conds, noise_type, noise_scale, max_len,
         total_iterations, n_cores, use_gpu):
    config = load_config(config_template)
    config['task']['metric'] = metric_name
    data_query_oracle = Equation_evaluator(equation_name, num_init_conds,
                                           noise_type, noise_scale,
                                           metric_name=metric_name)
    # print(data_query_oracle.vars_range_and_types_to_json)
    dataXgen = DataX(data_query_oracle.vars_range_and_types_to_json)
    nvars = data_query_oracle.get_nvars()
    function_set = data_query_oracle.get_operators_set()

    time_span = (0.0001, 2)
    trajectory_time_steps = 100
    max_opt_iter = 500
    t_eval = np.linspace(time_span[0], time_span[1], trajectory_time_steps)
    task = RegressTask(num_init_conds,
                       nvars,
                       dataXgen,
                       data_query_oracle,
                       time_span, t_eval)

    # get basic production rules
    reward_thresh = threshold_values[metric_name]
    nt_nodes, start_symbols = construct_non_terminal_nodes_and_start_symbols(nvars)
    production_rules = []
    for one_nt_node in nt_nodes:
        print(get_production_rules(nvars, function_set, one_nt_node))
        production_rules.extend(get_production_rules(nvars, function_set, one_nt_node))

    print("grammars:", production_rules)
    print("start_symbols:", start_symbols)
    program = grammarProgram(non_terminal_nodes=nt_nodes,
                             optimizer=optimizer,
                             metric_name=metric_name,
                             n_cores=n_cores,
                             max_opt_iter=max_opt_iter)
    grammar_model = ContextFreeGrammar(
        nvars=nvars,
        production_rules=production_rules,
        start_symbols=start_symbols,
        non_terminal_nodes=nt_nodes,
        max_length=max_len,
        hof_size=10,
        reward_threhold=reward_thresh
    )

    grammar_model.task = task
    grammar_model.program = program

    """Trains and returns dict of reward, expressions"""
    model = ActDeepSymbolicRegression(config, grammar_model)

    # Load training and test data
    X_constants, X_rnn, y_constants, y_rnn = get_data()

    # Establish GPU device if necessary
    if use_gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(use_gpu))
    else:
        device = torch.device("cpu")

    # Initialize operators, RNN, and optimizer


    max_length = 15
    type = 'lstm'
    num_layers = 2
    hidden_size = 250
    dropout = 0.0
    lr = 0.0005

    dsr_rnn = NeuralExpressionDecoder(hidden_size,
                                      max_length=max_length, cell=type, dropout=dropout, device=device).to(device)
    if optimizer == 'adam':
        optim = torch.optim.Adam(dsr_rnn.parameters(), lr=lr)
    else:
        optim = torch.optim.RMSprop(dsr_rnn.parameters(), lr=lr)
    # Perform the regression task
    results = learn(
        dsr_rnn,
        optim,
        inner_optimizer='rmsprop',
        inner_lr=0.1,
        inner_num_epochs=25,
        entropy_coefficient=0.005,
        risk_factor=0.95,
        initial_batch_size=2000,
        scale_initial_risk=True,
        batch_size=500,
        n_epochs=500,
        live_print=True,
        summary_print=True
    )

    # Unpack results
    epoch_best_rewards = results[0]
    epoch_best_expressions = results[1]
    best_reward = results[2]
    best_expression = results[3]

    # Plot best rewards each epoch
    print(epoch_best_rewards)
    print(epoch_best_expressions)
    print(best_reward)
    print(best_expression)






if __name__ == '__main__':
    main()
