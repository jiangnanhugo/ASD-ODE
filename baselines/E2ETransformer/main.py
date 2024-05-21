import torch
import click
import sympy as sp
import symbolicregression
from scibench.symbolic_data_generator import DataX, compute_time_derivative
import numpy as np
from grammar.grammar_regress_task import RegressTask
from scibench.symbolic_equation_evaluator import Equation_evaluator
from grammar.minimize_coefficients import execute
from sympy import lambdify, Symbol
import sys
import warnings

warnings.filterwarnings("ignore")


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
    for metric_name in dict_of_result:
        print(f"{metric_name} {dict_of_result[metric_name]}")
    print('-' * 30)


@click.command()
@click.option('--equation_name', default=None, type=str, help="Name of equation")
@click.option('--metric_name', default='inv_nrmse', type=str, help="evaluation metrics")
@click.option('--num_init_conds', default=10, type=int, help="batch of initial condition of dataset")
@click.option('--noise_type', default='normal', type=str, help="")
@click.option('--noise_scale', default=0.0, type=float, help="")
@click.option('--pretrained_model_filepath', type=str, help="pertrained pytorch model filepath")
@click.option('--mode', type=str, default='cpu', help="cpu or cuda")
def main(equation_name, metric_name, num_init_conds, noise_type, noise_scale, pretrained_model_filepath, mode):
    data_query_oracle = Equation_evaluator(equation_name,
                                           noise_type, noise_scale,
                                           metric_name=metric_name)
    dataX = DataX(data_query_oracle.vars_range_and_types_to_json)
    nvars = data_query_oracle.get_nvars()
    input_var_Xs = [Symbol(f'X{i}') for i in range(nvars)]
    time_span = (0.0001, 2)
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

    model = torch.load(pretrained_model_filepath, map_location=torch.device(mode))

    est = symbolicregression.model.SymbolicTransformerRegressor(
        model=model,
        max_input_points=200,
        n_trees_to_refine=100,
        rescale=True
    )
    topk = 5
    task.rand_draw_init_cond()
    print(f"PRINT Best Equations")
    print("=" * 20)
    for _ in range(topk):
        one_predict_ode = []
        for xi in range(nvars):
            est.fit(X_train.reshape(-1, nvars),
                    y_train[:, :, xi].reshape(-1, 1))

            replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/", 'nan': '1', 'inf': '1'}
            model_str = est.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix()

            for op, replace_op in replace_ops.items():
                model_str = model_str.replace(op, replace_op)
            for i in range(nvars):
                model_str = model_str.replace(f'x_{i}', f"X{i}")
            print(sp.parse_expr(model_str))
            one_predict_ode.append(model_str)
        temp = SymbolicDifferentialEquations(one_predict_ode)

        print_expressions(temp, task, input_var_Xs)
    print("=" * 20)


if __name__ == "__main__":
    main()
