# README: Active Discovery of Ordinary Differential Equations via Phase Portrait Sketching

## 1. Prerequisite of using these methods

- install the dependency package

```bash
pip install -r requirements.txt
```

## 2. Directory

### Dataset

- `data_oracle`: the generated dataset. Every file represents a ground-truth expression.
- `data_oracle/scibench`: the data oracle API to draw data. Before you run our program, you need to install the
  dataoracle by

```bash
cd data_oracle/scibench
pip install -e .
```

### Methods

- `apps_ode_pytorch`: the proposed method.
- `baselines/ProGED`: from https://github.com/brencej/ProGED.
- `baselines/SPL`: symbolic physics learner, from https://github.com/isds-neu/SymbolicPhysicsLearner.
- `baselines/E2E`: End to end transformer for symbolic regression, from https://github.com/facebookresearch/symbolicregression.
- `baselines/odeformer`: ODEFormer: Symbolic Regression of Dynamical Systems with Transformers.

#### Extra

- plots: the jupyter notebook to generate our figure.
- result: contains all the output of all the programs, the training logs.

### 3. Look at the summarized result
The experimental results are summarized in the `result` and `plots` folders.





