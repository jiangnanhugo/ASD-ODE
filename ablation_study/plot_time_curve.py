import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
from grammar.production_rules import get_production_rules, construct_non_terminal_nodes_and_start_symbols
from grammar.minimize_coefficients import execute

import matplotlib.pyplot as plt
from scibench.data import equation_object_loader
import os

palette = ['#ff3b30', '#4cd964', '#ffcc00', '#007aff', '#5856d6', '#ff9500', '#5ac8fa', '#ff2d55', '#969783']
markers = ['o', '^', '<', 'd', '*', '2']
sns.set_style("ticks")

mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['lines.linewidth'] = 1
rc("font", **{'family': 'serif', 'serif': ['Palatino'], 'size': 12})
rc('text', usetex=True)
plt.style.use("seaborn-v0_8-bright")

noise_type = 'normal'
noise_scale = 0.05
metric_name = 'neg_mse'
time_sequence_drop_rate = 0.5


def irregular_time_squence(true_trajectories, time_sequence, time_sequence_drop_rate,
                           ):
    random_mask = np.random.choice([0, 1],
                                   size=(true_trajectories.shape[0]),
                                   p=[time_sequence_drop_rate, 1 - time_sequence_drop_rate])

    # Apply the mask to the time steps
    masked_true_traj = true_trajectories[random_mask != 0]
    # masked_true_traj = true_trajectories * expanded_mask
    masked_time_sequence = time_sequence[random_mask != 0]
    # masked_time_sequence = time_sequence * random_mask
    return masked_true_traj, masked_time_sequence


def plot(equation_name, our_predict_eq, init_conds=None, title="", name="test"):
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
    trajectory_time_steps = 100
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
    if init_conds is not None:
        init_conds = task.rand_draw_init_cond()
    else:
        task.init_cond = init_conds
    true_trajectories = task.evaluate()[0]
    irregular_true_trajectories, irregular_time = irregular_time_squence(true_trajectories,
                                                                         t_eval,
                                                                         time_sequence_drop_rate)
    input_var_Xs = [Symbol(f'X{i}') for i in range(nvars)]
    pred_trajectories = execute(our_predict_eq, init_conds, time_span, t_eval, input_var_Xs)[0]
    print(pred_trajectories.shape)
    print(t_eval.shape)
    print(irregular_true_trajectories.shape)
    print(irregular_time.shape)
    plt.figure(figsize=(4, 3))
    for dim in range(nvars):
        plt.plot(t_eval, pred_trajectories[:, dim], color=f'C{dim}', alpha=.2, lw=10, label='predicted '+r"$x_{}$".format(dim))
        plt.plot(irregular_time, irregular_true_trajectories[:, dim], ls="None", marker='o', alpha=.3, color=f'C{dim}',
                 label='True '+r"$x_{}$".format(dim))
    plt.grid(False)
    plt.xlabel("Time Sequence (second)")
    plt.ylabel("Variable Value")
    plt.legend(loc='best')
    legend = plt.legend()

    # Remove the boundary and background color
    legend.get_frame().set_edgecolor('none')
    legend.get_frame().set_facecolor('none')
    # plt.show()
    # true_equation = equation_object_loader(name)
    # ranged = [xi.range for xi in true_equation.vars_range_and_types]
    # print(name, ranged, true_equation)
    # func = lambda x0, x1: true_equation.np_eq(t=None, x=[x0, x1]).tolist()
    #
    # fig = plt.figure(figsize=(3, 3))
    # fig, ax = Oscillator1.plot(color="grey")
    # # title = [r"\dot{x}_" + str(i + 1) + "=" + one_str for i, one_str in enumerate(true_equation.sympy_eq)]
    # title = f"Equation ID {i}"
    # # + ": " + true_equation._description
    # print(title)
    plt.title(title, fontsize=11)
    # # plt.show()
    # #
    fname = os.path.join(name + "_time_seq.pdf")
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()


name = "tes1"
plot('vars2_prog5', ['1.04*X1', '-0.02-0.77*X0'], np.array([3, -0.5]),
     title=r'$\phi=(1.04x_2, -0.02-0.77x_1)$',
     name=name)
name = 'test2'
plot('vars2_prog5', ['10*sin(X1)', '4*cos(X0+2)'], np.array([3, -0.5]),
     title=r'$\phi=(10\sin(x_2), 4\cos(x_1+2)$',
     name=name)
