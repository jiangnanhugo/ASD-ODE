
from odeformer.model import SymbolicTransformerRegressor
from odeformer.metrics import r2_score

from scipy.integrate import solve_ivp

import click
import numpy as np
from scibench.symbolic_equation_evaluator import Equation_evaluator
from scibench.symbolic_data_generator import DataX


# ignore user warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(1000)  # Seed for reproducibility
def get_data(func, nvars):
    t_train = np.linspace(0.0001, 10, 500)
    x0 = np.random.rand(nvars)
    x_train = solve_ivp(func, (t_train[0], t_train[-1]), x0, t_eval=t_train).y.T
    return t_train, x_train




def predict():



@click.command()
@click.option('--equation_name', default=None, type=str, help="Name of equation")
@click.option('--optimizer', default='BFGS', type=str, help="optimizer for the expressions")
@click.option('--metric_name', default='inv_nrmse', type=str, help="evaluation metrics")
@click.option('--num_init_conds', default=10, type=int, help="batch of initial condition of dataset")
@click.option('--noise_type', default='normal', type=str, help="")
@click.option('--noise_scale', default=0.0, type=float, help="")
def main(equation_name, metric_name, num_init_conds, noise_type, noise_scale):
    data_query_oracle = Equation_evaluator(equation_name, num_init_conds,
                                           noise_type, noise_scale,
                                           metric_name=metric_name)
    print(data_query_oracle.vars_range_and_types_to_json)
    dataXgen = DataX(data_query_oracle.vars_range_and_types_to_json)
    nvars = data_query_oracle.get_nvars()


    time_span = (0.0001, 2)
    trajectory_time_steps = 20

    t_eval = np.linspace(time_span[0], time_span[1], trajectory_time_steps)

    dstr = SymbolicTransformerRegressor(from_pretrained=True)

    model_args = {'beam_size': 200,
                  'beam_temperature': 0.1}
    dstr.set_model_args(model_args)

    t_train, x_train = get_data(data_query_oracle.true_equation,nvars)

    print(x_train.shape)
    print(t_train.shape)
    dstr.fit(t_train, x_train)

    dstr.print()

    pred_traj = dstr.predict(t_train, x_train[0])
    print(r2_score(x_train, pred_traj))


if __name__ == "__main__":
    main()

