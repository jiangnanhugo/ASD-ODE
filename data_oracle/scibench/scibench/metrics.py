import numpy as np
import scipy
from sklearn.metrics import r2_score

all_metrics = {
    # Negative mean squared error
    "neg_mse": lambda y, y_hat, var_y: -np.mean((y - y_hat) ** 2),
    # Negative root mean squared error
    "neg_rmse": lambda y, y_hat, var_y: -np.sqrt(np.mean((y - y_hat) ** 2)),
    # Negative normalized mean squared error
    "neg_nmse": lambda y, y_hat, var_y: -np.mean((y - y_hat) ** 2) / var_y,
    "log_nmse": lambda y, y_hat, var_y: -np.log10(1e-60 + np.mean((y - y_hat) ** 2) / var_y),
    # Negative normalized root mean squared error
    "neg_nrmse": lambda y, y_hat, var_y: -np.sqrt(np.mean((y - y_hat) ** 2) / var_y),
    # (Protected) inverse mean squared error
    "inv_mse": lambda y, y_hat, var_y: 1 / (1 + np.mean((y - y_hat) ** 2)),
    # (Protected) inverse normalized mean squared error
    "inv_nmse": lambda y, y_hat, var_y: 1 / (1 + np.mean((y - y_hat) ** 2) / var_y),
    # (Protected) inverse normalized root mean squared error
    "inv_nrmse": lambda y, y_hat, var_y: 1 / (1 + np.sqrt(np.mean((y - y_hat) ** 2) / var_y)),
    # Pearson correlation coefficient       # Range: [0, 1]
    "pearson": lambda y, y_hat, var_y: scipy.stats.pearsonr(y, y_hat)[0],
    # Spearman correlation coefficient      # Range: [0, 1]
    "spearman": lambda y, y_hat, var_y: scipy.stats.spearmanr(y, y_hat)[0],
    # Accuracy based on R2 value.
    "r2_score": lambda y, y_hat, var_y: r2_score(y.reshape(-1, y.shape[-1]), y_hat.reshape(-1, y_hat.shape[-1]),
                                                 multioutput='variance_weighted'),
}


def construct_noise(noise_type):
    """
    shape: multi dimensions.
    """
    _all_samplers = {
        'normal': lambda scale, shape: np.random.normal(loc=0.0, scale=scale, size=shape),
        'exponential': lambda scale, shape: np.random.exponential(scale=scale, size=shape),
        'uniform': lambda scale, shape: np.random.uniform(low=-np.abs(scale), high=np.abs(scale), size=shape),
        'laplace': lambda scale, shape: np.random.laplace(loc=0.0, scale=scale, size=shape),
        'logistic': lambda scale, shape: np.random.logistic(loc=0.0, scale=scale, size=shape)
    }
    assert noise_type in _all_samplers, "Unrecognized noise_type" + noise_type

    return _all_samplers[noise_type]
