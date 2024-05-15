import math
import os
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, odeint

warnings.filterwarnings("ignore")


# MODEL EQUATIONS
def VDP(t, x, p=[0.5]):
    return [1 * x[1],
            p[0] * (1 - x[0] ** 2) * x[1] - x[0]]


def RAY(t, x):
    return [1 * x[0] - 1 * x[1] - 1 * x[0] ** 3,
            1 * x[0]]


def POI(t, x):
    return [1 * x[0] - 1 * x[1] - 1 * x[0] * math.sqrt((x[0] ** 2) + (x[1] ** 2)),
            1 * x[0] + 1 * x[1] - 1 * x[1] * math.sqrt((x[0] ** 2) + (x[1] ** 2))]


def STL(t, x):
    return [1 * x[0] - 1 * x[1] - 1 * x[0] ** 3 - 1 * x[0] * x[1] ** 2,
            1 * x[0] + 1 * x[1] - 1 * x[1] ** 3 - 1 * x[1] * x[0] ** 2]


def lorenz(t, x, sigma=10, rho=28, beta=2.66667):
    return [
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ]


def lorenz_stable(t, x, sigma=10, rho=16, beta=8 / 3):
    return [
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ]


# yeast glycolysis model, note that there are many typos in the sindy-pi paper
def Glycolytic(
        t,
        x,
        c1=2.5,
        c2=-100,
        c3=13.6769,
        d1=200,
        d2=13.6769,
        d3=-6,
        d4=-6,
        e1=6,
        e2=-64,
        e3=6,
        e4=16,
        f1=64,
        f2=-13,
        f3=13,
        f4=-16,
        f5=-100,
        g1=1.3,
        g2=-3.1,
        h1=-200,
        h2=13.6769,
        h3=128,
        h4=-1.28,
        h5=-32,
        j1=6,
        j2=-18,
        j3=-100,
):
    return [
        c1 + c2 * x[0] * x[5] / (1 + c3 * x[5] ** 4),
        d1 * x[0] * x[5] / (1 + d2 * x[5] ** 4) + d3 * x[1] - d4 * x[1] * x[6],
        e1 * x[1] + e2 * x[2] + e3 * x[1] * x[6] + e4 * x[2] * x[5],
        f1 * x[2] + f2 * x[3] + f3 * x[4] + f4 * x[2] * x[5] + f5 * x[3] * x[6],
        g1 * x[3] + g2 * x[4],
        h3 * x[2]
        + h5 * x[5]
        + h4 * x[2] * x[6]
        + h1 * x[0] * x[5] / (1 + h2 * x[5] ** 4),
        j1 * x[1] + j2 * x[1] * x[6] + j3 * x[3] * x[6],
    ]


# Cart on a pendulum
def pendulum_on_cart(t, x, m=1, M=1, L=1, F=0, g=9.81):
    return [
        x[2],
        x[3],
        ((M + m) * g * np.sin(x[0]) - F * np.cos(x[0]) - m * L * np.sin(x[0]) * np.cos(x[0]) * x[2] ** 2) / (
                    L * (M + m * np.sin(x[0]) ** 2)),
        (m * L * np.sin(x[0]) * x[2] ** 2 + F - m * g * np.sin(x[0]) * np.cos(x[0])) / (M + m * np.sin(x[0]) ** 2),
    ]


def custom_func(t, x, sys_func):
    return [sys_func[i](*x) for i in range(len(sys_func))]


def custom_func_with_time(t, x, sys_func):
    return [sys_func[i](*x, t) for i in range(len(sys_func))]


# Carbone and Veltri triadic MHD model
def mhd(t, x, nu=0.0, mu=0.0, sigma=0.0):
    return [
        -2 * nu * x[0] + 4.0 * (x[1] * x[2] - x[4] * x[5]),
        -5 * nu * x[1] - 7.0 * (x[0] * x[2] - x[3] * x[5]),
        -9 * nu * x[2] + 3.0 * (x[0] * x[1] - x[3] * x[4]),
        -2 * mu * x[4] + 2.0 * (x[5] * x[1] - x[2] * x[4]),
        -5 * mu * x[4] + sigma * x[5] + 5.0 * (x[2] * x[3] - x[0] * x[5]),
        -9 * mu * x[5] + sigma * x[4] + 9.0 * (x[4] * x[0] - x[1] * x[3]),
    ]

# main function
def generate_ODE_data(system, inits, **generation_settings):
    generation_settings_preset = {
        "initial_time": 0,  # initial time
        "simulation_step": 0.01,  # simulation step /s
        "simulation_time": 50,  # simulation time (final time) /s
        "rtol": 1e-12,
        "atol": 1e-12,
        "method": 'LSODA',
        "custom_func_type": None,
        "custom_func": None}

    generation_settings_preset.update(generation_settings)
    generation_settings = generation_settings_preset

    t_span = [generation_settings["initial_time"], generation_settings["simulation_time"]]
    t = np.arange(generation_settings["initial_time"],
                  generation_settings["simulation_time"],
                  generation_settings["simulation_step"])

    if isinstance(system, str):
        system = eval(system)
    if generation_settings["custom_func_type"] == 'custom_func':
        system = lambda t, x: custom_func(t, x, generation_settings["custom_func"])
    elif generation_settings["custom_func_type"] == 'custom_func_with_time':
        system = lambda t, x: custom_func_with_time(t, x, generation_settings["custom_func"])

    """    
    X = solve_ivp(fun=system,
                  t_span=t_span,
                  y0=inits,
                  t_eval=t,
                  **generation_settings).y.T
    """

    X = odeint(system, inits, t,
               rtol=generation_settings["rtol"],
               atol=generation_settings["atol"],
               tfirst=True)

    return np.column_stack([t.reshape((len(t), 1)), X])


if __name__ == "__main__":

    # general settings for simulation
    generation_settings = {"initial_time": 0,
                           "simulation_step": 0.01,
                           "simulation_time": 50,
                           "method": 'LSODA',
                           "rtol": 1e-12,
                           "atol": 1e-12}

    ## 1. SIMPLE VERSION -- for single data set
    system = 'VDP'
    inits = [-0.2, -0.8]
    data = generate_ODE_data(system, inits, **generation_settings)

    ## 2. MORE COMPLEX DATA GENERATION -- for multiple datasets and multiple initial values
    ## the datasets are also saved in a file
    data_version = '1'
    data_path = "C:\\Users\\NinaO\\PycharmProjects\\ProGED\\data\\v{}\\".format(data_version)
    os.makedirs(data_path, exist_ok=True)

    systems = ['VDP', 'POI']
    num_system_variables = [2, 2, 3]
    num_inits = 10
    inits = np.random.uniform(low=-6, high=6, size=(num_inits, max(num_system_variables)))
    pd.DataFrame(inits).to_csv(data_path + 'inits.csv')

    for idx_init, iinit in enumerate(inits):
        for idx_system, isystem in enumerate(systems):
            data = generate_ODE_data(isystem, iinit[:num_system_variables[idx_system]], **generation_settings)
            os.makedirs(data_path + isystem + "\\", exist_ok=True)

            filename = '{}{}\\data_v{}_{}_init{}.csv'.format(data_path, isystem, data_version, isystem, str(idx_init))
            pd.DataFrame(data).to_csv(path_or_buf=filename, header=False, index=False)
