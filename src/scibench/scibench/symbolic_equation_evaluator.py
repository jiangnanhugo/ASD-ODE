import numpy as np

from scipy.integrate import solve_ivp

from sympy import lambdify, symbols
from scibench.metrics import all_metrics, construct_noise
from scibench.data import equation_object_loader

EQUATION_EXTENSION = ".in"


# def sympy_plus_scipy():
#     from sympy import symbols, lambdify
#     import numpy as np
#     import scipy.integrate
#     import matplotlib.pyplot as plt
#
#     # Create symbols y0, y1, and y2
#     y = symbols('y:3')
#
#     rf = y[0] ** 2 * y[1]
#     rb = y[2] ** 2
#     # Derivative of the function y(t); values for the three chemical species
#     # for input values y, kf, and kb
#     ydot = [2 * (rb - rf), rb - rf, 2 * (rf - rb)]
#     print(ydot)
#     t = symbols('t')  # not used in this case
#     # Convert the SymPy symbolic expression for ydot into a form that
#     # SciPy can evaluate numerically, f
#     f = lambdify((t, y), ydot)
#     k_vals = np.array([0.42, 0.17])  # arbitrary in this case
#     y0 = [1, 1, 0]  # initial condition (initial values)
#     t_eval = np.linspace(0, 10, 50)  # evaluate integral from t = 0-10 for 50 points
#     # Call SciPy's ODE initial value problem solver solve_ivp by passing it
#     #   the function f,
#     #   the interval of integration,
#     #   the initial state, and
#     #   the arguments to pass to the function f
#     solution = scipy.integrate.solve_ivp(f, (0, 10), y0, t_eval=t_eval)
#     # Extract the y (concentration) values from SciPy solution result
#     y = solution.y
#     # Plot the result graphically using matplotlib
#     plt.plot(t_eval, y.T)
#     # Add title, legend, and axis labels to the plot
#     plt.title('Chemical Kinetics')
#     plt.legend(['NO', 'Br$_2$', 'NOBr'], shadow=True)
#     plt.xlabel('time')
#     plt.ylabel('concentration')
#     # Finally, display the annotated plot
#     plt.show()


class Equation_evaluator(object):
    def __init__(self, true_equation_name, batch_size, noise_type='normal', noise_scale=0.0, metric_name="neg_nmse"):
        '''
        true_equation_name: the program name to map from X to Y
        noise_type, noise_scale: the type and scale of noise.
        metric_name: evaluation metric name for `y_true` and `y_pred`
        '''
        self.eq_name = true_equation_name
        self.batch_size = batch_size

        self.true_equation = equation_object_loader(true_equation_name)
        t = symbols('t')

        assert self.true_equation, "true_equation is not found"
        self.nvars = self.true_equation.num_vars
        self.operators_set = self.true_equation._operator_set
        self.vars_range_and_types = self.true_equation.vars_range_and_types
        self.input_var_Xs = self.true_equation.x
        self.true_ode_equation = lambdify((t, self.input_var_Xs), self.true_equation.sympy_eq)

        # metric
        self.metric_name = metric_name
        self.metric = all_metrics[metric_name]

        # noise
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.noises = construct_noise(self.noise_type)

    # def draw_init_condition(self):
    #     return self.true_equation.initialize(self.batch_size)

    def evaluate(self, x_init_conds: list, time_span: tuple, t_evals: np.ndarray) -> list:
        true_trajectories = []
        for one_x_init in x_init_conds:
            one_solution = solve_ivp(self.true_ode_equation, t_span=time_span, y0=one_x_init, t_eval=t_evals)
            true_trajectories.append(one_solution.y)
        return true_trajectories

    def _evaluate_loss(self, X_init_cond, time_span: tuple, t_evals: np.ndarray, pred_trajectories:np.ndarray)->float:
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

    def _evaluate_all_losses(self, X_init_cond,  time_span: tuple, t_evals: np.ndarray, pred_trajectories:np.ndarray):
        """
        Compute all the metrics between true_trajectories and pred_trajectories
        """
        pred_trajectories = pred_trajectories
        true_trajectories = self.evaluate(X_init_cond, time_span, t_evals)
        loss_val_dict = {}
        for metric_name in ['neg_nmse', 'neg_nrmse', 'inv_nrmse', 'inv_nmse','neg_mse', 'neg_rmse', 'inv_mse']:
            metric = all_metrics[metric_name]
            loss_val = self.metric(true_trajectories, pred_trajectories, np.var(true_trajectories))
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
