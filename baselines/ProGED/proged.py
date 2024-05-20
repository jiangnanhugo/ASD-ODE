import time
import pandas as pd
from ProGED.model_box import ModelBox
from ProGED.parameter_estimation import fit_models
from ProGED.utils.generate_data_ODE_systems import generate_ODE_data
from ProGED.configs import settings
from ProGED.equation_discoverer import EqDisco
from sympy import lambdify, Symbol
from grammar.minimize_coefficients import execute

import click
import numpy as np
from scibench.symbolic_equation_evaluator import Equation_evaluator
from scibench.symbolic_data_generator import DataX

# ignore user warnings
import warnings
from grammar.grammar_regress_task import RegressTask

warnings.filterwarnings("ignore", category=UserWarning)


# np.random.seed(1000)  # Seed for reproducibility
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


def proged_predict(system, nvars, inits, trajectory_time_steps):
    generation_settings = {
        "initial_time": 0,  # initial time
        "simulation_step": 0.001,  # simulation step /s
        "simulation_time": trajectory_time_steps,  # simulation time (final time) /s
    }

    data = generate_ODE_data(system=system, inits=inits, **generation_settings)
    all_symbols = [f'X{i}' for i in range(nvars)]
    all_d_symbols = [f'd_X{i}' for i in range(nvars)]
    data = pd.DataFrame(data, columns=['t'] + all_symbols)
    print(all_symbols)
    ED = EqDisco(data=data,
                 task_type="differential",
                 lhs_vars=all_d_symbols,
                 system_size=nvars,
                 rhs_vars=all_symbols,
                 generator="grammar",
                 generator_template_name="universal",
                 sample_size=100,
                 verbosity=1)

    ED.generate_models()

    models = ModelBox()
    for mi in ED.models:
        models.add_model([str(xi) for xi in mi.expr], symbols={"x": all_symbols, "const": "C"})

    settings['task_type'] = 'differential'
    settings["parameter_estimation"]["task_type"] = 'differential'
    settings["parameter_estimation"]["param_bounds"] = ((-5, 28),)
    settings["objective_function"]["persistent_homology"] = True

    weight = 0.70
    settings["objective_function"]["persistent_homology_weight"] = weight
    scale = 20

    settings["optimizer_DE"]["max_iter"] = 50 * scale
    settings["optimizer_DE"]["pop_size"] = scale
    settings["optimizer_DE"]["verbose"] = True

    start = time.time()
    models = fit_models(models, data, settings=settings)
    duration = time.time() - start
    print('duration seconds: ', duration)
    all_expressions = []
    for i in range(len(models)):
        params = list(models[i].params.values())
        print(params, f'{weight}')
        all_expressions.append(models[i].nice_print(return_string=True))

    return all_expressions


@click.command()
@click.option('--equation_name', default=None, type=str, help="Name of equation")
@click.option('--metric_name', default='inv_nrmse', type=str, help="evaluation metrics")
@click.option('--num_init_conds', default=10, type=int, help="batch of initial condition of dataset")
@click.option('--noise_type', default='normal', type=str, help="")
@click.option('--noise_scale', default=0.0, type=float, help="")
def main(equation_name, metric_name, num_init_conds, noise_type, noise_scale):
    data_query_oracle = Equation_evaluator(equation_name, batch_size=1,
                                           noise_type=noise_type, noise_scale=noise_scale,
                                           metric_name=metric_name)
    print(data_query_oracle.vars_range_and_types_to_json)
    nvars = data_query_oracle.get_nvars()
    input_var_Xs = [Symbol(f'X{i}') for i in range(nvars)]
    dataXgen = DataX(data_query_oracle.vars_range_and_types_to_json)

    time_span = (0.001, 1)
    trajectory_time_steps = 1000

    t_eval = np.linspace(time_span[0], time_span[1], trajectory_time_steps)
    task = RegressTask(1,
                       nvars,
                       dataXgen,
                       data_query_oracle,
                       time_span, t_eval)

    print('-' * 30)
    task.rand_draw_init_cond()
    x_init = task.init_cond
    all_pred_exprs = proged_predict(
        data_query_oracle.true_equation.np_eq,
        nvars,
        x_init.flatten(),
        trajectory_time_steps)
    topK=[]
    for one_expr in all_pred_exprs:
        pred_trajectories = execute(one_expr,
                                    x_init, task.time_span, task.t_evals,
                                    input_var_Xs)
        metric = task.evaluate_loss(pred_trajectories)
        topK.append((one_expr, metric))
    sorted_list = sorted(topK, key=lambda x: x[1])
    print(sorted_list)
    for one_pred_ode in all_pred_exprs:
        temp = SymbolicDifferentialEquations(one_pred_ode)

        print_expressions(temp, task, input_var_Xs)


if __name__ == "__main__":
    main()
