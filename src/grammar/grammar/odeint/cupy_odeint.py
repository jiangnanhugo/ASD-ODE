import time

import cupy as cp
import warnings



def euler_method(func, times, x_init):
    """
    https://perso.crans.org/besson/publis/notebooks/Runge-Kutta_methods_for_ODE_integration_in_Python.html
    """
    n = len(times)
    y = cp.zeros((n, len(x_init)))
    y[0] = x_init
    for i in range(len(times) - 1):
        y[i + 1] = y[i] + (times[i + 1] - times[i]) * cp.asarray(func(times[i], y[i]))
    return y


def runge_kutta4(func, times, x_init):
    """
    solve a batch of initial conditions
    """
    n = len(times)
    y = cp.zeros((n, len(x_init)))
    y[0] = x_init
    for i in range(len(times) - 1):
        h = times[i + 1] - times[i]
        k1 = cp.asarray(func(times[i], y[i]))
        k2 = cp.asarray(func(times[i] + h / 2., y[i] + k1 * h / 2))
        k3 = cp.asarray(func(times[i] + h / 2, y[i] + k2 * h / 2))
        k4 = cp.asarray(func(times[i] + h, y[i] + k3 * h))
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y

def cupy_sympy_plus_scipy():
    from sympy import symbols, lambdify
    import jax.numpy as jnp
    import scipy.integrate

    # Create symbols y0, y1, and y2
    y = symbols('y:3')

    rf = y[0] ** 2 * y[1]
    rb = y[2] ** 2
    # Derivative of the function y(t); values for the three chemical species
    # for input values y, kf, and kb
    ydot = [2 * (rb - rf), rb - rf, 2 * (rf - rb)]
    print(ydot)
    t = symbols('t')  # not used in this case
    # Convert the SymPy symbolic expression for ydot into a form that
    # SciPy can evaluate numerically, f
    f = lambdify((t, y), ydot, 'jax')

    y0 = [1, 0, 1]  # initial condition (initial values)
    y0 = jnp.asarray(y0)
    y0 = y0.T
    print(y0.shape)
    t_eval = jnp.linspace(0, 100, 5000)
    st = time.time()
    solution = scipy.integrate.solve_ivp(f, (0, 100), y0, t_eval=t_eval, method='LSODA')
    # Extract the y (concentration) values from SciPy solution result
    used_time = time.time() - st
    print("sympy+cupy+scipy used time", used_time)

def verify():
    from sympy import symbols, lambdify

    import scipy.integrate

    # Create symbols y0, y1, and y2
    y = symbols('y:3')

    # Derivative of the function y(t); values for the three chemical species
    # for input values y, kf, and kb
    def f(t, y):
        rf = y[0] ** 2 * y[1]
        rb = y[2] ** 2
        return [2 * (rb - rf), rb - rf, 2 * (rf - rb)]

    # print(ydot)
    t = symbols('t')  # not used in this case
    # Convert the SymPy symbolic expression for ydot into a form that
    # SciPy can evaluate numerically, f
    # f = lambdify((t, y), ydot)

    y0 = [1, 0, 1]  # initial condition (initial values)
    y0 = cp.asarray(y0)
    y0 = y0.T
    print(y0.shape)
    t_eval = cp.linspace(0, 100, 5000)
    st = time.time()
    solution = runge_kutta4(func=f, times=t_eval, x_init=y0)
    # Extract the y (concentration) values from SciPy solution result
    used_time = time.time() - st
    print("cupy used time", used_time)

    # Plot the result graphically using matplotlib

# verify()
cupy_sympy_plus_scipy()

