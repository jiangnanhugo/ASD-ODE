import time
import pandas as pd
import numpy as np
from ProGED.model_box import ModelBox
from ProGED.parameter_estimation import fit_models
from ProGED.utils.generate_data_ODE_systems import generate_ODE_data
from ProGED.configs import settings
from ProGED.equation_discoverer import EqDisco
np.random.seed(0)

print("dx = 10(y - x) \n"
      "dy = x(28 - z) - y \n"
      "dz = xy - 2.66667z")
generation_settings = {
    "initial_time": 0,  # initial time
    "simulation_step": 0.001,  # simulation step /s
    "simulation_time": 20,  # simulation time (final time) /s
}
data = generate_ODE_data(system='lorenz', inits=[0.2, 0.8, 0.5],
                         **generation_settings)
data = pd.DataFrame(data, columns=['t', 'x0', 'x1', 'x2'])

ED = EqDisco(data=data,
             task_type="differential",
             lhs_vars=["d_x0", "d_x1", "d_x2"],
             system_size=3,
             rhs_vars=["x0", "x1", "x2"],
             generator="grammar",
             generator_template_name="polynomial",
             generator_settings={
                 "p_S": [0.4, 0.6],
                 "p_T": [0.4, 0.6],
                 "p_vars": [0.33, 0.33, 0.34],
                 "p_R": [1, 0],
                 "p_F": [],
                 "functions": [],
             },
             sample_size=10,
             verbosity=4)

ED.generate_models()

models = ModelBox()
for mi in ED.models:
    models.add_model([str(xi) for xi in mi.expr],symbols={"x": ["x0", "x1", "x2"], "const": "C"})


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
print([10, 28, -2.66667])
for i in range(len(models)):
    params = list(models[i].params.values())
    print(params, f'{weight}')
    models[i].nice_print()
