import numpy as np
import scipy
from sklearn.metrics import r2_score
from program import execute

class symbolicRegressionTask(object):
    """
    used to handle input data 'X' for querying the data oracle.
    also used to set the controlled variables in input data `X`
    """

    def __init__(self, batchsize, n_input, X_train, y_train, metric_name='neg_nmse'):
        """
            batchsize: batch size
            allowed_input: 1 if the input variable is free. 0 if the input variable is controlled.
            dataX: generate the input data.
        """
        self.batchsize = batchsize
        self.n_input = n_input
        self.X_train = X_train
        self.y_train = y_train
        self.metric = all_metrics[metric_name]

    def rand_draw_X(self):
        self.selected_idx = np.random.choice(len(self.y_train), self.batchsize)
        self.X_fixed = self.X_train[self.selected_idx]
        self.y_fixed = self.y_train[self.selected_idx]
        return self.X_fixed, self.y_fixed

    def reward_function(self, p, input_vars):
        y_hat = execute(p, self.X_fixed, input_vars)

        var_ytrue = np.var(self.y_train)
        return self.metric(self.y_fixed, y_hat, var_ytrue)


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
