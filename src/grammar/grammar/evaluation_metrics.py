import numpy as np
import scipy




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
    "pearson": lambda y, y_hat: scipy.stats.pearsonr(y, y_hat)[0],
    # Spearman correlation coefficient      # Range: [0, 1]
    "spearman": lambda y, y_hat: scipy.stats.spearmanr(y, y_hat)[0],
    # Accuracy based on R2 value.
    "accuracy(r2)": lambda y, y_hat, var_y, tau: 1 - np.mean((y - y_hat) ** 2) / var_y >= tau,
}

