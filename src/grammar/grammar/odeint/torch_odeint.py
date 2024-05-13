import time

import torch
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

if torch.cuda.is_available():
    torch.set_default_device('cuda')


def runge_kutta4(func, times, x_init):
    """
    solve a batch of initial conditions
    """
    n = len(times)
    y = torch.zeros((n, len(x_init)))
    y[0] = x_init
    for i in range(len(times) - 1):
        h = times[i + 1] - times[i]
        k1 = torch.tensor(func(times[i], y[i]))
        k2 = torch.tensor(func(times[i] + h / 2., y[i] + k1 * h / 2))
        k3 = torch.tensor(func(times[i] + h / 2, y[i] + k2 * h / 2))
        k4 = torch.tensor(func(times[i] + h, y[i] + k3 * h))
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def f(t, y):
    rf = y[0] ** 2 * y[1]
    rb = y[2] ** 2
    return [2 * (rb - rf), rb - rf, 2 * (rf - rb)]


def verify():
    # print(ydot)

    y0 = [1, 0, 1]  # initial condition (initial values)
    y0 = torch.tensor(y0).T

    print(y0.shape)
    t_eval = torch.linspace(0, 100, 5000)
    st = time.time()
    solution = runge_kutta4(func=f, times=t_eval, x_init=y0)
    # Extract the y (concentration) values from SciPy solution result
    used_time = time.time() - st
    print("used time", used_time)

    # Plot the result graphically using matplotlib


if __name__ == '__main__':
    verify()
