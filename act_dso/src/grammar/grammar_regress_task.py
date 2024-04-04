import numpy as np


class RegressTask(object):
    """
    used to handle input data 'X' for querying the data oracle.
    also used to set the controlled variables in input data `X`
    """

    def __init__(self, dataset_size, n_vars, dataX, data_query_oracle, protected=False):
        """
            dataset_size: size of dataset.
            allowed_input: 1 if the input variable is free. 0 if the input variable is controlled.
            dataX: generate the input data.
            data_query_oracle: compute the output.
        """
        self.dataset_size = dataset_size

        self.dataX = dataX
        self.data_query_oracle = data_query_oracle
        #
        self.n_vars = n_vars

        
        self.protected = protected
        self.X_fixed = self.rand_draw_X_fixed()
        print(f"X_fixed: {self.X_fixed}")

    def rand_draw_X_non_fixed(self):
        self.X = self.dataX.randn(sample_size=self.dataset_size).T

    def rand_draw_X_fixed(self):
        self.X_fixed = np.squeeze(self.dataX.randn(sample_size=1))

    def rand_draw_X_fixed_with_index(self, xi):
        X_fixed = np.squeeze(self.dataX.randn(sample_size=1))
        self.X_fixed[xi] = X_fixed[xi]


    def rand_draw_data_with_X_fixed(self):
        self.X = self.dataX.randn(sample_size=self.dataset_size).T
        if self.X_fixed is None:
            self.rand_draw_X_fixed()

    def evaluate(self):
        return self.data_query_oracle.evaluate(self.X)

    def reward_function(self, y_hat):

        return self.data_query_oracle._evaluate_loss(self.X, y_hat)
