import numpy as np


class RegressTask(object):
    """
    used to handle the initial condition and the time span of the ODE for querying the proc_data oracle.
    """

    def __init__(self, num_init_conds, n_vars, dataX, data_query_oracle,
                 time_span=(0, 10), t_evals=np.linspace(0, 10, 50), num_of_regions=0, width=1):
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
        self.width = width

    def rand_draw_regions(self):
        """ draw several regions for each variable"""
        self.regions = self.dataX.rand_draw_regions(self.num_of_regions, self.width)
        return self.regions

    def full_init_cond(self, full_mesh_size):
        z = self.dataX.randn(sample_size=full_mesh_size)
        full_mesh = np.meshgrid(*[z[i] for i in range(self.n_vars)])
        self.full_mesh_init_cond = np.column_stack([ri.ravel() for ri in full_mesh])
        return self.full_mesh_init_cond

    def draw_init_cond(self):
        if self.init_cond is not None:
            return self.init_cond
        else:
            self.init_cond = self.dataX.randn(sample_size=self.num_init_conds).T
            return self.init_cond

    def rand_draw_init_cond(self, sample_size=None, one_region=None):
        if sample_size is None:
            sample_size = self.num_init_conds
        init_cond = self.dataX.randn(sample_size=sample_size, one_region=one_region).T
        self.init_cond = init_cond.reshape([-1, self.n_vars])
        return self.init_cond

    def evaluate(self):
        return self.data_query_oracle.evaluate(self.init_cond, self.time_span, self.t_evals)

    def evaluate_loss(self, pred_trajectories):
        return self.data_query_oracle._evaluate_loss(self.init_cond, self.time_span, self.t_evals, pred_trajectories)

    def evaluate_all_losses(self, pred_trajectories):
        return self.data_query_oracle._evaluate_all_losses(self.init_cond, self.time_span, self.t_evals,
                                                           pred_trajectories)
