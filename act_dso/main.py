import time
import click

from scibench.symbolic_data_generator import *
from scibench.symbolic_equation_evaluator_public import Equation_evaluator

from grammar.grammar import ContextFreeGrammar
from grammar.grammar_regress_task import RegressTask
from grammar.production_rules import get_production_rules, get_var_i_production_rules, construct_non_terminal_nodes_and_start_symbols
from grammar.grammar_program import grammarProgram
from deep_symbolic_optimizer import ActDeepSymbolicRegression
from utils import load_config, create_uniform_generations, create_reward_threshold

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
@click.option('--noise_type', default='normal', type=str, help="")
@click.option('--noise_scale', default=0.0, type=float, help="")
@click.option('--max_len', default=10, help="max length of the sequence from the decoder")
@click.option('--num_per_rounds', default=20, help="Number of iterations per rounds")
@click.option('--n_cores', default=1, help="Number of cores for parallel evaluation")
def main(config_template, optimizer, equation_name, metric_name, noise_type, noise_scale, max_len, num_per_rounds, n_cores):
    config = load_config(config_template)
    config['task']['metric'] = metric_name
    data_query_oracle = Equation_evaluator(equation_name, noise_type, noise_scale, metric_name=metric_name)
    dataXgen = DataX(data_query_oracle.get_vars_range_and_types())
    nvar = data_query_oracle.get_nvars()
    function_set = data_query_oracle.get_operators_set()

    num_iterations = create_uniform_generations(num_per_rounds, nvar)
    program = grammarProgram(optimizer=optimizer, metric_name=metric_name, n_cores=n_cores)

    regress_dataset_size = 2048
    task = RegressTask(regress_dataset_size,
                       nvar,
                       dataXgen,
                       data_query_oracle)

    # get basic production rules
    production_rules = get_production_rules(0, function_set)
    reward_thresh = create_reward_threshold(10, len(num_iterations))
    nt_nodes, start_symbols = construct_non_terminal_nodes_and_start_symbols(nvar)

    nt_nodes = ['A']
    start_symbols = ['A']
    best_expressions = None

    best_expressions_Q = []
    g_start = time.time()
    for round_idx in range(len(num_iterations)):
        print('++++++++++++ ROUND {}  ++++++++++++'.format(round_idx))

        if round_idx < len(num_iterations):
            production_rules += get_var_i_production_rules(round_idx, function_set)
        print("grammars:", production_rules)
        print("start_symbols:", start_symbols, nt_nodes[0] in start_symbols[0])

        grammar_model = ContextFreeGrammar(
            nvars=nvar,
            production_rules=production_rules,
            start_symbols=start_symbols[0],
            non_terminal_nodes=nt_nodes,
            max_length=max_len,
            hof_size=10,
            reward_threhold=reward_thresh[round_idx]
        )
        grammar_model.expr_obj_thres = threshold_values[metric_name]['expr_obj_thres']
        grammar_model.task = task
        grammar_model.program = program

        """Trains DSO and returns dict of reward, expressions"""
        model = ActDeepSymbolicRegression(config, grammar_model)
        start = time.time()
        print("deep model setup.....")

        model.setup()

        if nt_nodes[0] in start_symbols[0] and num_iterations[round_idx]:
            print("training starting......")
            best_expressions = model.train(
                threshold_values[metric_name]['reward_threshold'],
                num_iterations[round_idx]
            )
            end_time = time.time() - start

            print("dso time {} mins".format(np.round(end_time / 60, 3)))
            best_expressions_Q.extend(best_expressions)
        else:
            print("skipping training......")

        if round_idx < len(num_iterations) - 1:
            # the last round does not need freeze

            print(start_symbols)

            production_rules = [gi for gi in production_rules if str(round_idx) not in gi]

        grammar_model.print_hofs(verbose=True)
        best_expressions_Q = grammar_model.print_and_sort_global_Qs(best_expressions_Q)
    end_time = time.time() - g_start
    print("Final dso time {} mins".format(np.round(end_time / 60, 3)))


if __name__ == "__main__":
    main()
