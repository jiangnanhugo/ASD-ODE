import time

import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.set_printoptions(precision=4, linewidth=np.inf)


def euler_method(func, times, x_init):
    """
    https://perso.crans.org/besson/publis/notebooks/Runge-Kutta_methods_for_ODE_integration_in_Python.html
    """
    n = len(times)
    y = np.zeros((n, len(x_init)))
    y[0] = x_init
    for i in range(len(times) - 1):
        y[i + 1] = y[i] + (times[i + 1] - times[i]) * np.asarray(func(times[i], y[i]))
    return y


def runge_kutta4(func, times, x_init):
    """
    solve a batch of initial conditions
    """
    n = len(times)
    y = np.zeros((n, len(x_init)))
    y[0] = x_init
    for i in range(len(times) - 1):
        h = times[i + 1] - times[i]
        k1 = np.asarray(func(times[i], y[i]))
        k2 = np.asarray(func(times[i] + h / 2., y[i] + k1 * h / 2))
        k3 = np.asarray(func(times[i] + h / 2, y[i] + k2 * h / 2))
        k4 = np.asarray(func(times[i] + h, y[i] + k3 * h))
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def numpy_implementation():
    from sympy import symbols, lambdify
    import numpy as np
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
    f = lambdify((t, y), ydot, modules='numpy')
    k_vals = np.array([0.42, 0.17])  # arbitrary in this case
    y0 = [1, 0, 1]  # initial condition (initial values)
    y0 = np.asarray(y0)
    y0 = y0.T
    print(y0.shape)
    t_eval = np.linspace(0, 100, 5000)
    st = time.time()
    solution = runge_kutta4(func=f, times=t_eval, x_init=y0)
    # Extract the y (concentration) values from SciPy solution result
    used_time = time.time() - st
    print("numpy used time", used_time)

    # Plot the result graphically using matplotlib

def sympy_plus_scipy():
    from sympy import symbols, lambdify
    import numpy as np
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
    f = lambdify((t, y), ydot)
    k_vals = np.array([0.42, 0.17])  # arbitrary in this case
    y0 = [1, 0, 1]  # initial condition (initial values)
    y0 = np.asarray(y0)
    y0 = y0.T
    print(y0.shape)
    t_eval = np.linspace(0, 100, 5000)
    st = time.time()
    solution = scipy.integrate.solve_ivp(f, (0, 100), y0, t_eval=t_eval, method='LSODA')
    # Extract the y (concentration) values from SciPy solution result
    used_time = time.time() - st
    print("sympy+empty used time", used_time)
def sympy_numpy_plus_scipy():
    from sympy import symbols, lambdify
    import numpy as np
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
    f = lambdify((t, y), ydot, 'numpy')
    k_vals = np.array([0.42, 0.17])  # arbitrary in this case
    y0 = [1, 0, 1]  # initial condition (initial values)
    y0 = np.asarray(y0)
    y0 = y0.T
    print(y0.shape)
    t_eval = np.linspace(0, 100, 5000)
    st = time.time()
    solution = scipy.integrate.solve_ivp(f, (0, 100), y0, t_eval=t_eval, method='LSODA')
    # Extract the y (concentration) values from SciPy solution result
    used_time = time.time() - st
    print("sympy+numpy used time", used_time)

def sympy_cupy_plus_scipy():
    from sympy import symbols, lambdify
    import cupy as np
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
    f = lambdify((t, y), ydot, 'cupy')
    k_vals = np.array([0.42, 0.17])  # arbitrary in this case
    y0 = [1, 0, 1]  # initial condition (initial values)
    y0 = np.asarray(y0)
    y0 = y0.T
    print(y0.shape)
    t_eval = np.linspace(0, 100, 5000)
    st = time.time()
    solution = scipy.integrate.solve_ivp(f, (0, 100), y0, t_eval=t_eval, method='LSODA')
    # Extract the y (concentration) values from SciPy solution result
    used_time = time.time() - st
    print("sympy+cupy used time", used_time)

def sympy_jax_plus_scipy():
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
    k_vals = jnp.array([0.42, 0.17])  # arbitrary in this case
    y0 = [1, 0, 1]  # initial condition (initial values)
    y0 = np.asarray(y0)
    y0 = y0.T
    print(y0.shape)
    t_eval = jnp.linspace(0, 100, 5000)
    st = time.time()
    solution = scipy.integrate.solve_ivp(f, (0, 100), y0, t_eval=t_eval, method='LSODA')
    # Extract the y (concentration) values from SciPy solution result
    used_time = time.time() - st
    print("sympy+jax used time", used_time)


def numba_plus_scipy():
    from numbalsoda import lsoda_sig, lsoda, dop853
    from numba import njit, cfunc
    import numpy as np

    @cfunc(lsoda_sig)
    def rhs(t, y, dy, p):
        rf = y[0] ** 2 * y[1]
        rb = y[2] ** 2
        dy[0] = 2 * (rb - rf)
        dy[1] = rb - rf
        dy[2] = 2 * (rf - rb)

    funcptr = rhs.address  # address to ODE function
    u0 = np.array([1, 0, 1])
    data = np.array([1.0])
    t_eval = np.linspace(0, 100, 5000)

    # integrate with lsoda method
    st = time.time()
    usol, success = lsoda(funcptr, u0, t_eval, data=data)
    used_time = time.time() - st
    print("numba used time", used_time)


def symengine_plus_scipy():
    from symengine import symbols, Lambdify
    import numpy as np
    import scipy.integrate

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
    f = Lambdify((t, y[0], y[1], y[2]), ydot)

    def dx_dt(t, y):
        rf = y[0] ** 2 * y[1]
        rb = y[2] ** 2
        return [2 * (rb - rf), rb - rf, 2 * (rf - rb)]
        # return f(t, x, y, z)

    y0 = [1, 0, 1]  # initial condition (initial values)
    y0 = np.asarray(y0)
    y0 = y0.T
    print(y0.shape)
    t_eval = np.linspace(0, 100, 5000)
    st = time.time()
    solution = scipy.integrate.solve_ivp(dx_dt, (0, 100), y0, t_eval=t_eval, method='LSODA')
    used_time = time.time() - st
    print("symengine used time", used_time)


def sympy_vs_symengine():
    import time
    import numpy as np
    import sympy as sp
    import symengine as se
    import warnings
    x = sp.symarray('x', 14)
    p = sp.symarray('p', 14)
    args = np.concatenate((x, p))
    exp = sp.exp
    exprs = [x[0] + x[1] - x[4] + 36.252574322669, x[0] - x[2] + x[3] + 21.3219379611249,
             x[3] + x[5] - x[6] + 9.9011158998744, 2 * x[3] + x[5] - x[7] + 18.190422234653,
             3 * x[3] + x[5] - x[8] + 24.8679190043357, 4 * x[3] + x[5] - x[9] + 29.9336062089226,
             -x[10] + 5 * x[3] + x[5] + 28.5520551531262, 2 * x[0] + x[11] - 2 * x[4] - 2 * x[5] + 32.4401680272417,
             3 * x[1] - x[12] + x[5] + 34.9992934135095, 4 * x[1] - x[13] + x[5] + 37.0716199972041,
             p[0] - p[1] + 2 * p[10] + 2 * p[11] - p[12] - 2 * p[13] + p[2] + 2 * p[5] + 2 * p[6] + 2 * p[7] + 2 * p[
                 8] + 2 * p[9] - exp(x[0]) + exp(x[1]) - 2 * exp(x[10]) - 2 * exp(x[11]) + exp(x[12]) + 2 * exp(
                 x[13]) - exp(x[2]) - 2 * exp(x[5]) - 2 * exp(x[6]) - 2 * exp(x[7]) - 2 * exp(x[8]) - 2 * exp(x[9]),
             -p[0] - p[1] - 15 * p[10] - 2 * p[11] - 3 * p[12] - 4 * p[13] - 4 * p[2] - 3 * p[3] - 2 * p[4] - 3 * p[
                 6] - 6 * p[7] - 9 * p[8] - 12 * p[9] + exp(x[0]) + exp(x[1]) + 15 * exp(x[10]) + 2 * exp(
                 x[11]) + 3 * exp(x[12]) + 4 * exp(x[13]) + 4 * exp(x[2]) + 3 * exp(x[3]) + 2 * exp(x[4]) + 3 * exp(
                 x[6]) + 6 * exp(x[7]) + 9 * exp(x[8]) + 12 * exp(x[9]),
             -5 * p[10] - p[2] - p[3] - p[6] - 2 * p[7] - 3 * p[8] - 4 * p[9] + 5 * exp(x[10]) + exp(x[2]) + exp(
                 x[3]) + exp(x[6]) + 2 * exp(x[7]) + 3 * exp(x[8]) + 4 * exp(x[9]),
             -p[1] - 2 * p[11] - 3 * p[12] - 4 * p[13] - p[4] + exp(x[1]) + 2 * exp(x[11]) + 3 * exp(x[12]) + 4 * exp(
                 x[13]) + exp(x[4]),
             -p[10] - 2 * p[11] - p[12] - p[13] - p[5] - p[6] - p[7] - p[8] - p[9] + exp(x[10]) + 2 * exp(x[11]) + exp(
                 x[12]) + exp(x[13]) + exp(x[5]) + exp(x[6]) + exp(x[7]) + exp(x[8]) + exp(x[9])]

    lmb_sp = sp.lambdify(args, exprs, modules='numpy')
    lmb_se = se.Lambdify(args, exprs)
    lmb_se_cse = se.Lambdify(args, exprs, cse=True)
    lmb_se_llvm = se.Lambdify(args, exprs, backend='llvm')

    inp = np.ones(28)

    lmb_sp(*inp)
    tim_sympy = time.time()
    for i in range(500):
        res_sympy = lmb_sp(*inp)
    tim_sympy = time.time() - tim_sympy

    lmb_se(inp)
    tim_se = time.time()
    res_se = np.empty(len(exprs))
    for i in range(500):
        res_se = lmb_se(inp)
    tim_se = time.time() - tim_se

    lmb_se_cse(inp)
    tim_se_cse = time.time()
    res_se_cse = np.empty(len(exprs))
    for i in range(500):
        res_se_cse = lmb_se_cse(inp)
    tim_se_cse = time.time() - tim_se_cse

    lmb_se_llvm(inp)
    tim_se_llvm = time.time()
    res_se_llvm = np.empty(len(exprs))
    for i in range(500):
        res_se_llvm = lmb_se_llvm(inp)
    tim_se_llvm = time.time() - tim_se_llvm

    print(
        'SymEngine (lambda double)    vs sympy: {} {}'.format(tim_se, tim_sympy))

    print(
        'symengine (lambda double + CSE)  vs sympy:{} {}'.format(tim_se_cse, tim_sympy))

    print('symengine (LLVM)    vs sympy: {} {}'.format(tim_se_llvm, tim_sympy))

    import itertools
    from functools import reduce
    from operator import mul

    def ManualLLVM(inputs, *outputs):
        outputs_ravel = list(itertools.chain(*outputs))
        cb = se.Lambdify(inputs, outputs_ravel, backend="llvm")

        def func(*args):
            result = []
            n = np.empty(len(outputs_ravel))
            t = cb.unsafe_real(np.concatenate([arg.ravel() for arg in args]), n)
            start = 0
            for output in outputs:
                elems = reduce(mul, output.shape)
                result.append(n[start:start + elems].reshape(output.shape))
                start += elems
            return result

        return func

    lmb_se_llvm_manual = ManualLLVM(args, np.array(exprs))
    lmb_se_llvm_manual(inp)
    tim_se_llvm_manual = time.time()
    res_se_llvm_manual = np.empty(len(exprs))
    for i in range(500):
        res_se_llvm_manual = lmb_se_llvm_manual(inp)
    tim_se_llvm_manual = time.time() - tim_se_llvm_manual
    print('symengine (ManualLLVM) vs sympy: {} {}'.format(tim_se_llvm_manual, tim_sympy,
                                                          ))

    if tim_se_llvm_manual < tim_se_llvm:
        warnings.warn("Cython code for Lambdify.__call__ is slow.")


if __name__ == '__main__':
    # sympy_cupy_plus_scipy()
    sympy_numpy_plus_scipy()
    sympy_jax_plus_scipy()
    sympy_plus_scipy()
    symengine_plus_scipy()
    # numpy_implementation()
    # sympy_vs_symengine()


    numba_plus_scipy()
