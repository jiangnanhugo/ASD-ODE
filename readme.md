# README: Active Symbolic Discovery of Ordinary Differential Equations

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

- `act_dso`: the proposed method.
- `ProGED`: from https://github.com/brencej/ProGED.
- `SPL`: symbolic physics learner, from https://github.com/isds-neu/SymbolicPhysicsLearner.
- `E2E`: End to end transformer for symbolic regression, from https://github.com/facebookresearch/symbolicregression.
- `gp_and_cvgp`: genetic programming  (GP) and VSR-GP algorithm, from https://github.com/jiangnanhugo/cvgp
- `dso_classic`: the codebase for DSR, VPG, PQT and GPMeld, from https://github.com/dso-org/deep-symbolic-optimization
- `odeformer`: ODEFormer: Symbolic Regression of Dynamical Systems with Transformers.

#### Extra

- plots: the jupyter notebook to generate our figure.
- result: contains all the output of all the programs, the training logs.

### 3. Look at the summarized result
The experimental results are summarized in the `result` and `plots` folders.





