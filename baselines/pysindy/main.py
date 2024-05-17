import numpy as np
from scipy.integrate import solve_ivp
from pysindy.utils import lorenz

import pysindy as ps

# ignore user warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import torch
import click
from scibench.symbolic_data_generator import DataX
import numpy as np
from grammar.grammar_regress_task import RegressTask
from scibench.symbolic_equation_evaluator import Equation_evaluator
from grammar.minimize_coefficients import execute
from sympy import Symbol
import sys


class SymbolicDifferentialEquations(object):
    """
    this class is used to represent symbolic ordinary differential equations.
    For n variables settings, there will be n total expressions.
    """

    def __init__(self, list_of_sympy_equations):
        # self.traversal = list_of_rules
        self.fitted_eq = list_of_sympy_equations
        self.invalid = False
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
    data_query_oracle = Equation_evaluator(equation_name, num_init_conds,
                                           noise_type, noise_scale,
                                           metric_name=metric_name)
    dataX = DataX(data_query_oracle.vars_range_and_types_to_json)
    nvars = data_query_oracle.get_nvars()
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
    optimizer = ps.STLSQ(threshold=1e-6)

    library = ps.PolynomialLibrary(degree=2)
    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=library
    )

    model.fit(true_trajectories, t_eval)
    predicted_eq = model.equations()
    predicted_eq = [model_str.replace(' + ', '+') for model_str in predicted_eq]
    predicted_eq = [model_str.replace('+-', '-') for model_str in predicted_eq]
    predicted_eq = [model_str.replace(' ', '*') for model_str in predicted_eq]
    predicted_eq = [model_str.replace('^', '**') for model_str in predicted_eq]

    for i in range(nvars):
        predicted_eq = [model_str.replace(f'x{i}', f"X{i}") for model_str in predicted_eq]

    model.print()

    sys.stdout.flush()


    task.rand_draw_init_cond()
    print(f"PRINT Best Equations")
    print("=" * 20)

    temp = SymbolicDifferentialEquations(predicted_eq)
    print(temp)
    print_expressions(temp, task, input_var_Xs)
    print("=" * 20)


if __name__ == "__main__":
    main()
