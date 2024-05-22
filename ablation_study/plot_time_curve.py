
import matplotlib as mpl
import matplotlib.pyplot as plt
from sympy import *
import seaborn as sns
from matplotlib import rc

import torch
import time
import click
import numpy as np
from scibench.symbolic_equation_evaluator import Equation_evaluator
from scibench.symbolic_data_generator import DataX

from sympy import Symbol, simplify
from grammar.grammar_regress_task import RegressTask
from grammar.minimize_coefficients import execute

import matplotlib.pyplot as plt
from scibench.symbolic_data_generator import irregular_time_sequence
import os

palette = ['#ff3b30', '#4cd964', '#ffcc00', '#007aff', '#5856d6', '#ff9500', '#5ac8fa', '#ff2d55', '#969783']
markers = ['o', '^', '<', 'd', '*', '2']
sns.set_style("ticks")

mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['lines.linewidth'] = 1
rc("font", **{'family': 'serif', 'serif': ['Palatino'], 'size': 14})
rc('text', usetex=True)
plt.style.use("seaborn-v0_8-bright")

noise_type = 'normal'
noise_scale = 0.1
metric_name = 'neg_mse'
time_sequence_drop_rate = 0.5


def plot(equation_name, our_predict_eq, init_conds=None, title="", name="test", loc='lower right', ylabel=False):
    # equation_name = f"vars2_prog{eq_id}"
    data_query_oracle = Equation_evaluator(equation_name,
                                           noise_type, noise_scale,
                                           metric_name=metric_name,
                                           time_sequence_drop_rate=time_sequence_drop_rate)
    # print(data_query_oracle.vars_range_and_types_to_json)
    dataXgen = DataX(data_query_oracle.vars_range_and_types_to_json)
    nvars = data_query_oracle.get_nvars()
    # function_set = data_query_oracle.get_operators_set()

    time_span = (0.0, 10)
    trajectory_time_steps = 300
    num_init_conds = 1
    t_eval = np.linspace(time_span[0], time_span[1], trajectory_time_steps)
    num_regions = 10
    task = RegressTask(num_init_conds,
                       nvars,
                       dataXgen,
                       data_query_oracle,
                       time_span, t_eval,
                       num_of_regions=num_regions,
                       width=0.1)
    if init_conds is None:
        init_conds = task.rand_draw_init_cond()
    else:
        # print(task.rand_draw_init_cond().shape)
        # print(init_conds.shape)
        init_conds = init_conds.reshape(1, nvars)
        task.init_cond = init_conds
    true_trajectories = task.evaluate()[0]
    irregular_true_trajectories, irregular_time = irregular_time_sequence(true_trajectories,
                                                                          t_eval,
                                                                          time_sequence_drop_rate)
    input_var_Xs = [Symbol(f'X{i}') for i in range(nvars)]
    pred_trajectories = execute(our_predict_eq, init_conds, time_span, t_eval, input_var_Xs)[0]
    print(pred_trajectories.shape)
    print(t_eval.shape)
    print(irregular_true_trajectories.shape)
    print(irregular_time.shape)
    plt.figure(figsize=(3.5, 3.5))
    for dim in range(nvars):
        plt.plot(t_eval, pred_trajectories[:, dim], color=palette[1], alpha=.8, lw=3,
                 label='Predicted ' + r"$x_{}$".format(dim + 1))
        plt.plot(irregular_time, irregular_true_trajectories[:, dim], ls="None", marker='.', alpha=.8,
                 color=palette[3],
                 label='True ' + r"$x_{}$".format(dim + 1))
    plt.grid(False)
    plt.xlabel("Time Sequence (second)", fontsize=16)
    if ylabel == True:
        plt.ylabel("Variable Value", fontsize=16)
    # plt.legend(loc='upper right')
    legend = plt.legend(loc=loc)

    # Remove the boundary and background color
    # legend.get_frame().set_edgecolor('none')
    legend.get_frame().set_facecolor('none')
    plt.title(title, fontsize=16)
    # # plt.show()
    # #
    fname = os.path.join(name + "_time_seq.pdf")
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()


name = "odeformer.at01"
plot('vars2_prog5', ['1.04*X1', '-0.02-0.77*X0'], np.array([0, 1]),
     title='ODEFormer Prediction\n' + r'$\phi=(1.04x_2, -0.02-0.77x_1)$',
     name=name)
name = 'apps.at01'
plot('vars2_prog5', ['X1', '-0.895*sin(X0)'], np.array([0, 1]),
     title='APPS (our) Prediction\n' + r'$\phi=(x_2, 0.895\sin(x_1))$',
     name=name, ylabel=True)

name = "odeformer.at4neg1"
plot('vars2_prog5', ['1.04*X1', '-0.02-0.77*X0'], np.array([4, -1]),
     title='ODEFormer Prediction\n' + r'$\phi=(1.04x_2, -0.02-0.77x_1)$',
     name=name, loc='lower left', )
name = 'apps.at4neg1'
plot('vars2_prog5', ['X1', '-0.895*sin(X0)'], np.array([4, -1]),
     title='APPS (our) Prediction\n' + r'$\phi=(x_2, 0.895\sin(x_1))$',
     name=name, loc='lower left', )
