import numpy as np
import autograd.numpy as anp
from autograd import grad
from scipy.optimize import minimize
import time


# Define the Ackley function
def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    sum_sq_term = -a * anp.exp(-b * anp.sqrt(anp.sum(x ** 2) / n))
    cos_term = -anp.exp(anp.sum(anp.cos(c * x) / n))
    return a + anp.exp(1) + sum_sq_term + cos_term


# Calculate the gradient of the Ackley function using autograd
ackley_gradient = grad(ackley)

# Initial guess
x0 = np.array([0.5, 0.5])

# Record the start time
start_time = time.time()

# Optimize the Ackley function with gradient information
result = minimize(ackley, x0, jac=ackley_gradient, method='BFGS')

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print("Optimal solution:", result.x)
print("Optimal value:", result.fun)
print("Time used:", elapsed_time, "seconds")
st = time.time()
# Optimize the Rosenbrock function with gradient information
result = minimize(ackley, x0, method='BFGS')
used2 = time.time() - st
print("Optimal2 time:", used2)
print(used2 / elapsed_time)
# %%%
# even slower
import jax.numpy as jnp
from jax import grad, jit
from scipy.optimize import minimize
import time


# Define the Ackley function
def ackley(x):
    a = 20
    b = 0.2
    c = 2 * jnp.pi
    n = len(x)
    sum_sq_term = -a * jnp.exp(-b * jnp.sqrt(jnp.sum(x ** 2) / n))
    cos_term = -jnp.exp(jnp.sum(jnp.cos(c * x) / n))
    return a + jnp.exp(1) + sum_sq_term + cos_term


# Define the gradient of the Ackley function using JAX's grad function
ackley_gradient = grad(ackley)

# Initial guess
x0 = jnp.array([0.5, 0.5])

# Record the start time
start_time = time.time()

# Optimize the Ackley function with gradient information
result = minimize(ackley, x0, jac=ackley_gradient, method='BFGS')

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print("Optimal solution:", result.x)
print("Optimal value:", result.fun)
print("Time used:", elapsed_time, "seconds")
# %%%
from nelder_mead import nelder_mead
import numpy as np
from numba import njit
import time


# Define the Ackley function
@njit
def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / n))
    cos_term = -np.exp(np.sum(np.cos(c * x) / n))
    return a + np.exp(1) + sum_sq_term + cos_term


# Calculate the gradient of the Ackley function using autograd
ackley_gradient = grad(ackley)

# Initial guess
x0 = np.array([0.5, 0.5])

# Record the start time
start_time = time.time()

# Optimize the Ackley function with gradient information
result = nelder_mead(ackley, x0)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print("Final time used:", elapsed_time, "seconds")
