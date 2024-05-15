import time
import pandas as pd
import numpy as np
from ProGED.model_box import ModelBox
from ProGED.parameter_estimation import fit_models
from ProGED.utils.generate_data_ODE_systems import generate_ODE_data
from ProGED.configs import settings
from ProGED.equation_discoverer import EqDisco

np.random.seed(0)

nvars = 6
generation_settings = {
    "initial_time": 0,  # initial time
    "simulation_step": 0.001,  # simulation step /s
    "simulation_time": 10,  # simulation time (final time) /s
}

data = generate_ODE_data(system='mhd', inits=np.random.randn(nvars),
                         **generation_settings)
data = pd.DataFrame(data, columns=['t', 'x0', 'x1', 'x2', 'x3', 'x4', 'x5',])

ED = EqDisco(data=data,
             task_type="differential",
             lhs_vars=["d_x0", "d_x1", "d_x2", 'd_x3', 'd_x4', 'd_x5'],
             system_size=nvars,
             rhs_vars=["x0", "x1", "x2", 'x3', 'x4', 'x5'],
             generator="grammar",
             generator_template_name="universal",
             sample_size=100,
             verbosity=1)

ED.generate_models()

models = ModelBox()
for mi in ED.models:
    models.add_model([str(xi) for xi in mi.expr], symbols={"x": ["x0", "x1", "x2", 'x3', 'x4', 'x5'], "const": "C"})

settings['task_type'] = 'differential'
settings["parameter_estimation"]["task_type"] = 'differential'
settings["parameter_estimation"]["param_bounds"] = ((-5, 28),)
settings["objective_function"]["persistent_homology"] = True

weight = 0.70
settings["objective_function"]["persistent_homology_weight"] = weight
scale = 20

settings["optimizer_DE"]["max_iter"] = 50 * scale
settings["optimizer_DE"]["pop_size"] = scale
settings["optimizer_DE"]["verbose"] = True

start = time.time()
models = fit_models(models, data, settings=settings)
duration = time.time() - start
print(weight)
for i in range(len(models)):
    params = list(models[i].params.values())
    print(params, f'{weight}')
    models[i].nice_print()
