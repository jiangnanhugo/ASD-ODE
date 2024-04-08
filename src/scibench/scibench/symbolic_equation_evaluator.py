from scipy.integrate import solve_ivp
from scibench.metrics import all_metrics, construct_noise
import numpy as np
from scibench.data import equation_object_loader

EQUATION_EXTENSION = ".in"


class Equation_evaluator(object):
    def __init__(self, true_equation_name, batch_size, noise_type='normal', noise_scale=0.0, metric_name="neg_nmse"):
        '''
        true_equation_filename: the program to map from X to Y
        noise_type, noise_scale: the type and scale of noise.
        metric_name: evaluation metric name for `y_true` and `y_pred`
        '''
        self.eq_name = true_equation_name
        self.batch_size = batch_size

        self.true_equation = equation_object_loader(true_equation_name)
        assert self.true_equation, "true_equation is not found"
        self.nvars = self.true_equation.num_vars
        self.dt = self.true_equation.dt
        self.operators_set = self.true_equation._operator_set
        self.input_var_Xs = self.true_equation.x
        # self.specified_input_var_index= self.true_equation.specified_x_idx

        # metric
        self.metric_name = metric_name
        self.metric = all_metrics[metric_name]

        # noise
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.noises = construct_noise(self.noise_type)

    def draw_init_condition(self, initialize_type):
        return self.true_equation.initialize(self.batch_size, initialize_type)

    def execute_simulate(self, c0, return_last_step=True):
        c_new = self.true_equation.simulate_exec(c0)
        return c_new

    def _evaluate_simulate_loss(self, X_init, y_pred, simulate_steps:int=1):
        """
        Compute the y_true based on the init_cond. And then evaluate the metric value between y_true and y_pred
        """
        y_pred = y_pred
        y_true = self.execute_simulate(X_init, simulate_steps)

        if self.metric_name in ['neg_nmse', 'neg_nrmse', 'inv_nrmse', 'inv_nmse', 'neg_mse', 'neg_rmse', 'inv_mse']:
            loss_val = self.metric(y_true, y_pred, np.var(y_true))
        else:
            raise NotImplementedError(self.metric_name, "is not implemented....")
        return loss_val

    def _evaluate_simulate_all_losses(self, X_init, y_pred, simulate_steps):
        """
        Compute the y_true based on the input X. And then evaluate the metric value between y_true and y_pred
        """
        y_pred = y_pred
        y_true = self.execute_simulate(X_init, simulate_steps)
        loss_val_dict = {}
        for metric_name in ['neg_nmse', 'neg_nrmse', 'inv_nrmse', 'inv_nmse']:
            metric = all_metrics[metric_name]
            loss_val = metric(y_true, y_pred, np.var(y_true))
            loss_val_dict[metric_name] = loss_val
        for metric_name in ['neg_mse', 'neg_rmse', 'inv_mse']:
            metric = all_metrics[metric_name]
            loss_val = metric(y_true, y_pred)
            loss_val_dict[metric_name] = loss_val
        return loss_val_dict

    def _get_eq_name(self):
        return self.eq_name

    def get_nvars(self):
        return self.nvars

    def get_operators_set(self):
        return self.operators_set