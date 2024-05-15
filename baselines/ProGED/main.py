import time
import argparse
import random
import numpy as np
from scibench.symbolic_data_generator import DataX
from scibench.symbolic_equation_evaluator_public import Equation_evaluator
import pandas as pd

from ProGED import EqDisco
from ProGED.model_box import ModelBox
from ProGED.parameter_estimation import fit_models
from ProGED.configs import settings

def run_ProGED(
        data,
        nvars,
        generator_template_name
):
    ED = EqDisco(data=data,
                 task_type="algebraic",
                 lhs_vars=["y"],
                 # list the variable on the left-hand side of each equation in the system of equations, should match columns names in data
                 rhs_vars=[f"X{i}" for i in range(nvars)],
                 # list the variables that can appear on the right-hand side of each equation, should match columns names in data
                 sample_size=10,  # number of candidate equations to generate
                 generator="grammar",  # optional, accepts instance of custom generator or grammar
                 generator_template_name=generator_template_name,
                 # name of grammar template if not using custom generator, common choices: polynomial, rational, universal
                 verbosity=4)  # level of detail the program prints, 0 is silent besides warnings
    print(ED.generate_models())

    models = ModelBox()
    for mi in ED.models:
        models.add_model([str(xi) for xi in mi.expr], symbols={"x": [f"X{i}" for i in range(nvars)], "const": "C"})

    settings['task_type'] = 'algebraic'
    settings["parameter_estimation"]["task_type"] = 'algebraic'
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
    print(weight)
    for i in range(len(models)):
        params = list(models[i].params.values())
        print(params, f'{weight}')
        models[i].nice_print()

    # print(ED.fit_models())
    # print(ED.get_results())
    # print(ED.get_stats())



def ProGED(equation_name, metric_name, noise_type, noise_scale):
    data_query_oracle = Equation_evaluator(equation_name, noise_type, noise_scale, metric_name)
    dataX = DataX(data_query_oracle.get_vars_range_and_types())
    nvars = data_query_oracle.get_nvars()
    regress_batchsize = 25600
    operators_set = data_query_oracle.get_operators_set()
    X_train = dataX.randn(sample_size=regress_batchsize).T
    print(X_train.shape)
    y_train = data_query_oracle.evaluate(X_train)
    val_dict = {}
    for i in range(nvars):
        val_dict[f"X{i}"] = X_train[:, i]
    val_dict['y'] = y_train
    data = pd.DataFrame(val_dict)
    start = time.time()
    run_ProGED(data, nvars, generator_template_name='universal')
    end_time = time.time() - start

    print("ProGED {} mins".format(np.round(end_time / 60, 3)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--equation_name", help="the filename of the true program.")
    parser.add_argument('--optimizer',
                        nargs='?',
                        choices=['BFGS', 'L-BFGS-B', 'Nelder-Mead', 'CG', 'basinhopping', 'dual_annealing', 'shgo', 'direct'],
                        help='list servers, storage, or both (default: %(default)s)')
    parser.add_argument("--metric_name", type=str, default='neg_mse', help="The name of the metric for loss.")
    parser.add_argument("--noise_type", type=str, default='normal', help="The name of the noises.")
    parser.add_argument("--noise_scale", type=float, default=0.0, help="This parameter adds the standard deviation of the noise")

    args = parser.parse_args()

    seed = int(time.perf_counter() * 10000) % 1000007
    random.seed(seed)
    print('random seed=', seed)

    seed = int(time.perf_counter() * 10000) % 1000007
    np.random.seed(seed)
    print('np.random seed=', seed)
    print(args)

    ProGED(args.equation_name, args.metric_name, args.noise_type, args.noise_scale)
