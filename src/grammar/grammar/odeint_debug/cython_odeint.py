# https://github.com/jrenaud90/

import time
import numpy as np
from CyRK import nbrk_ode
from numba import njit


@njit
def rhs(t, y):
    rf = y[0] ** 2 * y[1]
    rb = y[2] ** 2
    dy = np.empty_like(y)
    dy[0] = 2 * (rb - rf)
    dy[1] = rb - rf
    dy[2] = 2 * (rf - rb)
    return dy


time_span = (0., 1000.)
u0 = np.array([1, 0, 1])

t_eval = np.linspace(0, 100, 5000)

st = time.time()
time_domain, y_results, success, message = nbrk_ode(rhs, time_span, u0, rk_method=2, t_eval=t_eval)
used_time = time.time() - st
print("numba used time", used_time)
# still very slow