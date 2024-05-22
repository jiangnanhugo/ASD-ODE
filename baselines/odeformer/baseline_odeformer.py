import time

from odeformer.model import SymbolicTransformerRegressor
from odeformer.metrics import r2_score

from scipy.integrate import solve_ivp

import click
import numpy as np
from scibench.symbolic_equation_evaluator import Equation_evaluator
from scibench.symbolic_data_generator import DataX
import scipy
# ignore user warnings
import warnings
from grammar.grammar_regress_task import RegressTask
from scibench.symbolic_data_generator import irregular_time_sequence

warnings.filterwarnings("ignore", category=UserWarning)

# np.random.seed(1000)  # Seed for reproducibility


all_metrics = {
    # Negative mean squared error
    "neg_mse": lambda y, y_hat, var_y: -np.mean((y - y_hat) ** 2),
    # Negative root mean squared error
    "neg_rmse": lambda y, y_hat, var_y: -np.sqrt(np.mean((y - y_hat) ** 2)),
    # Negative normalized mean squared error
    "neg_nmse": lambda y, y_hat, var_y: -np.mean((y - y_hat) ** 2) / var_y,
    "log_nmse": lambda y, y_hat, var_y: -np.log10(1e-60 + np.mean((y - y_hat) ** 2) / var_y),
    # Negative normalized root mean squared error
    "neg_nrmse": lambda y, y_hat, var_y: -np.sqrt(np.mean((y - y_hat) ** 2) / var_y),
    # (Protected) inverse mean squared error
    "inv_mse": lambda y, y_hat, var_y: 1 / (1 + np.mean((y - y_hat) ** 2)),
    # (Protected) inverse normalized mean squared error
    "inv_nmse": lambda y, y_hat, var_y: 1 / (1 + np.mean((y - y_hat) ** 2) / var_y),
    # (Protected) inverse normalized root mean squared error
    "inv_nrmse": lambda y, y_hat, var_y: 1 / (1 + np.sqrt(np.mean((y - y_hat) ** 2) / var_y)),
    # Pearson correlation coefficient       # Range: [0, 1]
    "pearson": lambda y, y_hat, var_y: scipy.stats.pearsonr(y, y_hat)[0],
    # Spearman correlation coefficient      # Range: [0, 1]
    "spearman": lambda y, y_hat, var_y: scipy.stats.spearmanr(y, y_hat)[0],
    # Accuracy based on R2 value.
    "r2_score": lambda y, y_hat, var_y: r2_score(y.reshape(-1, y.shape[-1]), y_hat.reshape(-1, y_hat.shape[-1]),
                                                 multioutput='variance_weighted'),
}


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
@click.option('--num_init_conds', default=50, type=int, help="batch of initial condition of dataset")
@click.option('--noise_type', default='normal', type=str, help="")
@click.option('--noise_scale', default=0.0, type=float, help="")
@click.option('--time_sequence_drop_rate', default=0, type=float, help="simulate irregular time sequence")
def main(pretrain_basepath, equation_name, metric_name, num_init_conds, noise_type, noise_scale,
         time_sequence_drop_rate):
    data_query_oracle = Equation_evaluator(equation_name,
                                           noise_type=noise_type, noise_scale=noise_scale,
                                           metric_name=metric_name,
                                           time_sequence_drop_rate=time_sequence_drop_rate)
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
    x_train, t_train = irregular_time_sequence(x_train,
                                               t_eval,
                                               time_sequence_drop_rate)
    print("after irregular time sequence:",x_train.shape, t_train.shape)
    # print("train trajectory data:")
    # for xi in x_train:
    #     print("[", ", ".join(map(str, xi)), "],")
    # print('-' * 30)
    st = time.time()
    dstr.fit(t_train, x_train)

    dstr.print(n_predictions=10)

    pred_traj = dstr.predict(t_train, x_train[0])
    used = time.time() - st
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
    all_true_traj = np.asarray(all_true_traj)
    all_pred_traj = np.asarray(all_pred_traj)
    print('-' * 30)
    for metric_name in ['neg_nmse', 'neg_nrmse', 'inv_nrmse', 'inv_nmse', 'neg_mse', 'neg_rmse', 'inv_mse']:
        print(metric_name, all_metrics[metric_name](all_true_traj, all_pred_traj, np.var(all_true_traj)))
    print('-' * 30)
    print('Total ODEFormer time:', used)


if __name__ == "__main__":
    main()
