import time
import click
import numpy as np
from scibench.symbolic_equation_evaluator import Equation_evaluator
from scibench.symbolic_data_generator import DataX

from grammar.grammar import ContextFreeGrammar
from grammar.grammar_regress_task import RegressTask
from grammar.production_rules import get_production_rules, construct_non_terminal_nodes_and_start_symbols
from grammar.grammar_program import grammarProgram
from deep_symbolic_optimizer import ActDeepSymbolicRegression
from utils import load_config

averaged_var_y = 10
threshold_values = {
    'neg_mse': {'reward_threshold': 1e-6, 'expr_obj_thres': 0.01},
    'neg_nmse': {'reward_threshold': 1e-6, 'expr_obj_thres': 0.01 / averaged_var_y},
    'neg_nrmse': {'reward_threshold': 1e-3, 'expr_obj_thres': np.sqrt(0.01 / averaged_var_y)},
    'neg_rmse': {'reward_threshold': 1e-3, 'expr_obj_thres': 0.1},
    'inv_mse': {'reward_threshold': 1 / (1 + 1e-6), 'expr_obj_thres': -1 / (1 + 1e-6)},
    'inv_nmse': {'reward_threshold': 1 / (1 + 1e-6), 'expr_obj_thres': -1 / (1 + 1e-6)},
    'inv_nrmse': {'reward_threshold': 1 / (1 + 1e-6), 'expr_obj_thres': -1 / (1 + 1e-6)},
}


@click.command()
@click.argument('config_template', default="")
@click.option('--equation_name', default=None, type=str, help="Name of equation")
@click.option('--optimizer', default='BFGS', type=str, help="optimizer for the expressions")
@click.option('--metric_name', default='inv_nrmse', type=str, help="evaluation metrics")
@click.option('--batch_size', default=10, type=int, help="batch of initial condition of dataset")
@click.option('--noise_type', default='normal', type=str, help="")
@click.option('--noise_scale', default=0.0, type=float, help="")
@click.option('--max_len', default=10, help="max length of the sequence from the decoder")
@click.option('--total_iterations', default=20, help="Number of iterations per rounds")
@click.option('--n_cores', default=1, help="Number of cores for parallel evaluation")
def main(config_template, optimizer, equation_name, metric_name, batch_size, noise_type, noise_scale, max_len,
         total_iterations,
         n_cores):
    config = load_config(config_template)
    config['task']['metric'] = metric_name
    data_query_oracle = Equation_evaluator(equation_name, batch_size, noise_type, noise_scale, metric_name=metric_name)
    print(data_query_oracle.vars_range_and_types_to_json)
    dataXgen = DataX(data_query_oracle.vars_range_and_types_to_json)
    nvars = data_query_oracle.get_nvars()
    function_set = data_query_oracle.get_operators_set()

    batch_size_of_trajectories = 20
    time_span = (0.0001, 10)
    trajectory_time_steps = 500
    t_eval = np.linspace(time_span[0], time_span[1], trajectory_time_steps)
    task = RegressTask(batch_size_of_trajectories,
                       nvars,
                       dataXgen,
                       data_query_oracle,
                       time_span, t_eval)

    # get basic production rules
    reward_thresh = 10
    nt_nodes, start_symbols = construct_non_terminal_nodes_and_start_symbols(nvars)
    production_rules = []
    for one_nt_node in nt_nodes:
        production_rules.extend(get_production_rules(nvars, function_set, one_nt_node))

    print("grammars:", production_rules)
    print("start_symbols:", start_symbols, nt_nodes[0] in start_symbols[0])
    program = grammarProgram(non_terminal_nodes=nt_nodes,
                             optimizer=optimizer,
                             metric_name=metric_name,
                             n_cores=n_cores)
    grammar_model = ContextFreeGrammar(
        nvars=nvars,
        production_rules=production_rules,
        start_symbols=start_symbols,
        non_terminal_nodes=nt_nodes,
        max_length=max_len,
        hof_size=10,
        reward_threhold=reward_thresh
    )

    grammar_model.expr_obj_thres = threshold_values[metric_name]['expr_obj_thres']
    grammar_model.task = task
    grammar_model.program = program

    """Trains DSO and returns dict of reward, expressions"""
    model = ActDeepSymbolicRegression(config, grammar_model)
    start = time.time()
    print("deep model setup.....")

    model.setup()

    print("training starting......")
    model.train(
        threshold_values[metric_name]['reward_threshold'],
        total_iterations
    )
    end_time = time.time() - start

    grammar_model.print_hofs(verbose=True)

    print("Final act_dso time {} mins".format(np.round(end_time / 60, 3)))


if __name__ == "__main__":
    main()
