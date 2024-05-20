import numpy as np

from scipy.integrate import solve_ivp
from scibench.solve_init_value_problem import runge_kutta4, runge_kutta2, euler_method
from sympy import lambdify, symbols
from scibench.metrics import all_metrics, construct_noise
from scibench.data import equation_object_loader

EQUATION_EXTENSION = ".in"


class Equation_evaluator(object):
    def __init__(self, true_equation_name, noise_type='normal', noise_scale=0.0, metric_name="neg_nmse"):
        '''
        true_equation_name: the program name to map from X to Y
        noise_type, noise_scale: the type and scale of noise.
        metric_name: evaluation metric name for `y_true` and `y_pred`
        '''
        self.eq_name = true_equation_name


        self.true_equation = equation_object_loader(true_equation_name)
        print("name:", self.true_equation._eq_name)
        print("description:", self.true_equation._description)
        print("operator_set:", self.true_equation._operator_set)
        print("expressions:", self.true_equation.sympy_eq)
        assert self.true_equation, "true_equation is not found"
        self.nvars = self.true_equation.num_vars
        self.operators_set = self.true_equation._operator_set
        self.vars_range_and_types = self.true_equation.vars_range_and_types
        self.vars_range_and_types_to_json = self.true_equation.vars_range_and_types_to_json_str()
        self.input_var_Xs = self.true_equation.x
        #
        self.true_ode_equation = self.true_equation.np_eq
        # metric
        self.metric_name = metric_name
        self.metric = all_metrics[metric_name]

        # noise
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.noises = construct_noise(self.noise_type)

    def evaluate(self, x_init_conds: list, time_span: tuple, t_evals: np.ndarray) -> np.ndarray:
        """
        compute the true trajectory for each x_init_cond
        """
        true_trajectories = []
        for one_x_init in x_init_conds:
            # one_solution = solve_ivp(self.true_ode_equation, t_span=time_span, t_evals=t_evals, y0=one_x_init,
            #                          method='RK45')
            # true_trajectories.append(one_solution.y)
            one_solution = runge_kutta4(self.true_ode_equation, t_evals, one_x_init)
            true_trajectories.append(one_solution)
        true_trajectories = np.asarray(true_trajectories)

        return true_trajectories + self.noises(self.noise_scale, true_trajectories.shape)

    def _evaluate_loss(self, X_init_cond, time_span: tuple, t_evals: np.ndarray,
                       pred_trajectories: np.ndarray) -> float:
        """
        1. Compute the true_trajectories based on the init_cond.
        2. evaluate the metric value between true_trajectories and pred_trajectories
        X_init_cond: [batch_size, nvars]
        pred_trajectories: [batch_size, time_steps, nvars]
        """
        pred_trajectories = pred_trajectories
        true_trajectories = self.evaluate(X_init_cond, time_span, t_evals)

        if self.metric_name in ['neg_nmse', 'neg_nrmse', 'inv_nrmse', 'inv_nmse', 'neg_mse', 'neg_rmse', 'inv_mse']:
            loss_val = self.metric(true_trajectories, pred_trajectories, np.var(true_trajectories))
        else:
            raise NotImplementedError(self.metric_name, "is not implemented....")
        return loss_val

    def _evaluate_all_losses(self, X_init_cond, time_span: tuple, t_evals: np.ndarray, pred_trajectories: np.ndarray):
        """
        Compute all the metrics between true_trajectories and pred_trajectories
        """
        pred_trajectories = pred_trajectories
        true_trajectories = self.evaluate(X_init_cond, time_span, t_evals)
        loss_val_dict = {}
        for metric_name in ['neg_mse', 'inv_mse', 'neg_nmse', 'inv_nmse', 'neg_nrmse', 'inv_nrmse',
                            'neg_rmse']:
            metric = all_metrics[metric_name]
            loss_val = metric(true_trajectories, pred_trajectories, np.var(true_trajectories))
            loss_val_dict[metric_name] = loss_val
        return loss_val_dict

    def _get_eq_name(self):
        return self.eq_name

    def get_nvars(self):
        """
        return the number of variables
        """
        return self.nvars

    def get_operators_set(self):
        """
        return the set of math operators
        """
        return self.operators_set
