import numpy as np


def init_solve_ivp(method):
    if method == 'euler':
        solve_ivp = euler_method
    elif method == 'RK23':
        solve_ivp = runge_kutta2
    elif method == 'RK45':
        solve_ivp = runge_kutta4
    else:
        raise NotImplementedError("Method {} not implemented".format(method))
    return solve_ivp


def solve_ivp(solve_ivp, func, times, x_init: np.ndarray) -> np.ndarray:
    """
    func: the function of ODE.
    times: discrete time points.
    x_init: initial values of ODE.
    """
    return solve_ivp(func, times, x_init)


def euler_method(func, times, x_init):
    """
    https://perso.crans.org/besson/publis/notebooks/Runge-Kutta_methods_for_ODE_integration_in_Python.html
    """
    n = len(times)
    y = np.zeros((n, len(x_init)))
    y[0] = x_init
    for i in range(len(times) - 1):
        y[i + 1] = y[i] + (times[i + 1] - times[i]) * func(times[i], y[i])
    return y


def runge_kutta2(func, times, x_init):
    n = len(times)
    y = np.zeros((n, len(x_init)))
    y[0] = x_init
    for i in range(len(times) - 1):
        h = times[i + 1] - times[i]
        k1 = func(times[i], y[i])
        y[i + 1] = y[i] + h * func(times[i] + h / 2, y[i] + k1 * h / 2)
    return y


def runge_kutta4(func, times, x_init):
    n = len(times)
    y = np.zeros((n, len(x_init)))
    y[0] = x_init
    for i in range(len(times) - 1):
        h = times[i + 1] - times[i]
        k1 = func(times[i], y[i])
        k2 = func(times[i] + h / 2., y[i] + k1 * h / 2)
        k3 = func(times[i] + h / 2, y[i] + k2 * h / 2)
        k4 = func(times[i] + h, y[i] + k3 * h)
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def compare_with_scipy_solve_ivp():
    from scipy.integrate import solve_ivp
    def lotkavolterra(t, z):
        x, y = z
        a, b, c, d = 1.5, 1, 3, 1
        return np.array([a * x - b * x * y, -c * y + d * x * y])

    t = np.linspace(0, 5, 5000)
    sol = solve_ivp(lotkavolterra, [0, 5], [10, 5], t_eval=t,
                    method='RK45')
    print(sol)

    output = runge_kutta4(lotkavolterra, t, [10, 5])
    print(output)

    import matplotlib.pyplot as plt

    plt.plot(t, sol.y.T)
    plt.plot(t, output)
    plt.xlabel('t')
    plt.legend(['x1', 'y1', 'x2', 'y2'], shadow=True)
    plt.title('Lotka-Volterra System')
    plt.show()
