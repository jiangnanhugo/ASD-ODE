import numpy as np


class RegressTask(object):
    """
    used to handle the initial condition and the time span of the ODE for querying the proc_data oracle.
    """

    def __init__(self, num_init_conds, n_vars, dataX, data_query_oracle,
                 time_span=(0, 10), t_evals=np.linspace(0, 10, 50), num_of_regions=-1):
        """
            n_vars: number of variables
            dataX: draw the initial conditions
            data_query_oracle: generate the time trajectory sequence
            time_span: total time of the trajectory.
            t_evals: the list of discrete time where the trajectory is evaluated.
        """
        self.num_init_conds = num_init_conds
        self.time_span = time_span
        self.t_evals = t_evals
        self.dataX = dataX
        self.data_query_oracle = data_query_oracle
        #
        self.n_vars = n_vars
        self.num_of_regions = num_of_regions

    def rand_draw_regions(self, num_of_regions):
        """ draw several regions for each variable"""
        self.regions = self.dataX.rand_draw_regions(num_of_regions)
        return self.regions

    def active_region_init_cond(self, ):
        init_cond = self.dataX.randn(sample_size=self.num_init_conds).T
        self.init_cond = init_cond.reshape([-1, self.n_vars])

    def full_init_cond(self):
        pass

    def rand_draw_init_cond(self):
        init_cond = self.dataX.randn(sample_size=self.num_init_conds).T
        self.init_cond = init_cond.reshape([-1, self.n_vars])

    def evaluate(self):
        return self.data_query_oracle.evaluate(self.init_cond, self.time_span, self.t_evals)

    def evaluate_loss(self, pred_trajectories):
        return self.data_query_oracle._evaluate_loss(self.init_cond, self.time_span, self.t_evals, pred_trajectories)

    def evaluate_all_losses(self, pred_trajectories):
        return self.data_query_oracle._evaluate_all_losses(self.init_cond, self.time_span, self.t_evals,
                                                           pred_trajectories)


# given a set of ODEs expressions, determine some trajecotry where most ODEs disagreee
# 1. randomly sample seveal regions and sketch a phase portrait of each small region.
# 2. sample some initial condition in each region
from scipy.optimize import fsolve
import numpy as np

from sympy import Matrix, Symbol, nonlinsolve


def sketch_phase_portraits(one_ode: list, list_of_regions, input_var_Xs: list):
    """

    https://stackoverflow.com/questions/49553006/compute-the-jacobian-matrix-in-python
    example:
    Matrix(['2*u1 + 3*u2','2*u1 - 3*u2']).jacobian(['u1', 'u2'])
    """
    # 1. find_fixed_points
    num_of_regions = len(list_of_regions)
    total_trajectories = 10000
    num_init_cond_each_region = total_trajectories // num_of_regions
    init_conds=[]
    for i in range(num_of_regions):
        for j in range(num_init_cond_each_region):
            init_conds.append(draw_init_cond(list_of_regions[i]))
