import numpy as np


class RegressTask(object):
    """
    used to handle the initial condition and the time span of the ODE for querying the proc_data oracle.
    """

    def __init__(self, num_init_conds, n_vars, dataX, data_query_oracle,
                 time_span=(0, 10), t_evals=np.linspace(0, 10, 50), num_of_regions=0):
        """
            n_vars: number of variables
            dataX: draw the initial conditions
            data_query_oracle: generate the time trajectory sequence
            time_span: total time of the trajectory.
            t_evals: the list of discrete time where the trajectory is evaluated.
            num_of_regions: used for actively sample regions from the phase portrait.
        """
        self.num_init_conds = num_init_conds
        self.time_span = time_span
        self.t_evals = t_evals
        self.dataX = dataX
        self.data_query_oracle = data_query_oracle
        #
        self.n_vars = n_vars
        self.num_of_regions = num_of_regions

    def rand_draw_regions(self):
        """ draw several regions for each variable"""
        self.regions = self.dataX.rand_draw_regions(self.num_of_regions)
        return self.regions

    def active_region_init_cond(self):
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





