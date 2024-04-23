import numpy as np
from ode_models import *
from odeformer.model import SymbolicTransformerRegressor
from odeformer.metrics import r2_score

from scipy.integrate import solve_ivp


# ignore user warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(1000)  # Seed for reproducibility
def get_lorenz_data():
    t_train = np.linspace(0.0001, 10, 500)
    x0 = np.random.rand(3)
    x_train = solve_ivp(lorenz, (t_train[0], t_train[-1]), x0, t_eval=t_train).y.T
    return t_train, x_train


def get_glycolysis_data():
    t_train = np.linspace(0.0001, 10, 500)
    x0 = np.random.rand(7)
    x_train = solve_ivp(glycolysis, (t_train[0], t_train[-1]), x0, t_eval=t_train).y.T
    return t_train, x_train


def get_mhd_data():
    t_train = np.linspace(0.0001, 10, 500)
    x0 = np.random.rand(6)
    x_train = solve_ivp(mhd, (t_train[0], t_train[-1]), x0, t_eval=t_train).y.T
    return t_train, x_train



dstr = SymbolicTransformerRegressor(from_pretrained=True)

model_args = {'beam_size': 200,
              'beam_temperature': 0.1}
dstr.set_model_args(model_args)

t_train, x_train = get_lorenz_data()

print(x_train.shape)
print(t_train.shape)
dstr.fit(t_train, x_train)

dstr.print()

pred_traj = dstr.predict(t_train, x_train[0])
print(r2_score(x_train, pred_traj))

t_train, x_train = get_mhd_data()

print(x_train.shape)
print(t_train.shape)
dstr.fit(t_train, x_train)

dstr.print()

pred_traj = dstr.predict(t_train, x_train[0])
print(r2_score(x_train, pred_traj))

dstr = SymbolicTransformerRegressor(from_pretrained=False)

model_args = {'beam_size': 200,
              'beam_temperature': 0.1}
dstr.set_model_args(model_args)

t_train, x_train = get_glycolysis_data()

print(x_train.shape)
print(t_train.shape)
dstr.fit(t_train, x_train)

dstr.print()

pred_traj = dstr.predict(t_train, x_train[0])
print(r2_score(x_train, pred_traj))
