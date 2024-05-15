

import numpy as np
from scipy.integrate import solve_ivp

# yeast glycolysis model, note that there are many typos in the sindy-pi paper
def yeast(
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


import pysindy as ps

# ignore user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(1000)  # Seed for reproducibility

t_train = np.linspace(0.0001, 10, 25600)
x0 = np.random.rand(7)
x_train = solve_ivp(yeast, (t_train[0], t_train[-1]), x0, t_eval=t_train).y.T

print(t_train)
print(x_train)
optimizer = ps.STLSQ(threshold=1e-6)
library = ps.PolynomialLibrary(degree=4)
model = ps.SINDy(
    optimizer=optimizer,
    feature_library=library
)
model.fit(x_train, t_train)
model.print()