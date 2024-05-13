from odeformer.model import SymbolicTransformerRegressor
from odeformer.metrics import r2_score

from scipy.integrate import solve_ivp

import click
import numpy as np
from scibench.symbolic_equation_evaluator import Equation_evaluator
from scibench.symbolic_data_generator import DataX

# ignore user warnings
import warnings
from grammar.grammar_regress_task import RegressTask

warnings.filterwarnings("ignore", category=UserWarning)

# np.random.seed(1000)  # Seed for reproducibility


def get_data(func, nvars, t_eval, task):
    task.rand_draw_init_cond()
    x0 = task.init_cond.flatten()

    print(x0.shape, x0)
    x_train = solve_ivp(func, (t_eval[0], t_eval[-1]), x0, t_eval=t_eval).y.T
    return t_eval, x_train


@click.command()
@click.option('--pretrain_basepath', default=None, type=str, help="folder of the pretrained model")
@click.option('--equation_name', default=None, type=str, help="Name of equation")
@click.option('--metric_name', default='inv_nrmse', type=str, help="evaluation metrics")
@click.option('--num_init_conds', default=10, type=int, help="batch of initial condition of dataset")
@click.option('--noise_type', default='normal', type=str, help="")
@click.option('--noise_scale', default=0.0, type=float, help="")
def main(pretrain_basepath, equation_name, metric_name, num_init_conds, noise_type, noise_scale):
    data_query_oracle = Equation_evaluator(equation_name, batch_size=1,
                                           noise_type=noise_type, noise_scale=noise_scale,
                                           metric_name=metric_name)
    print(data_query_oracle.vars_range_and_types_to_json)
    dataXgen = DataX(data_query_oracle.vars_range_and_types_to_json)

    nvars = data_query_oracle.get_nvars()

    time_span = (0.0001, 5)
    trajectory_time_steps = 500

    t_eval = np.linspace(time_span[0], time_span[1], trajectory_time_steps)
    task = RegressTask(1,
                       nvars,
                       dataXgen,
                       data_query_oracle,
                       time_span, t_eval)
    dstr = SymbolicTransformerRegressor(from_pretrained=True, pretrain_basepath=pretrain_basepath)

    model_args = {'beam_size': 200,
                  'beam_temperature': 0.1}
    dstr.set_model_args(model_args)

    t_train, x_train = get_data(data_query_oracle.true_equation.np_eq, nvars, t_eval, task)

    print(x_train.shape, t_train.shape)
    print("train trajectory data:")
    for xi in x_train:
        print("[", ", ".join(map(str, xi)), "],")
    print('-' * 30)
    # print("time sequence", t_train)
    dstr.fit(t_train, x_train)

    dstr.print(n_predictions=10)

    pred_traj = dstr.predict(t_train, x_train[0])
    one_r2_score = r2_score(np.asarray(pred_traj).flatten(), np.asarray(x_train).flatten())
    print("train R2 score:", one_r2_score)

    print('-' * 30)
    all_true_traj = []
    all_pred_traj = []
    for _ in range(num_init_conds):
        t_valid, x_valid = get_data(data_query_oracle.true_equation.np_eq, nvars, t_eval, task)
        pred_traj = dstr.predict(t_valid, x_valid[0])
        all_true_traj.append(x_valid)
        all_pred_traj.append(pred_traj)
        one_r2_score = r2_score(np.asarray(all_pred_traj).flatten(), np.asarray(all_true_traj).flatten())
        print("R2 score:", one_r2_score)
        print("neg_nmse:", -(1 - one_r2_score))
        print('-' * 30)


if __name__ == "__main__":
    main()
