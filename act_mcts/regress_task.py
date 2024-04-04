import numpy as np


class RegressTask(object):
    """
    used to handle input data 'X' for querying the data oracle.
    also used to set the controlled variables in input data `X`
    """

    def __init__(self, batchsize, allowed_input, dataX, data_query_oracle):
        """
            batchsize: batch size
            allowed_input: 1 if the input variable is free. 0 if the input variable is controlled.
            dataX: generate the input data.
        """
        self.batchsize = batchsize
        self.allowed_input = allowed_input
        self.n_input = allowed_input.size
        self.dataX = dataX
        self.data_query_oracle = data_query_oracle

        self.fixed_column = [i for i in range(self.n_input) if self.allowed_input[i] == 0]
        self.X_fixed = np.random.rand(self.n_input)

    def set_allowed_inputs(self, allowed_inputs):
        self.allowed_input = np.copy(allowed_inputs)
        self.fixed_column = [i for i in range(self.n_input) if self.allowed_input[i] == 0]

    def set_allowed_input(self, i, flag):
        self.allowed_input[i] = flag
        self.fixed_column = [i for i in range(self.n_input) if self.allowed_input[i] == 0]

    def rand_draw_X_non_fixed(self):
        self.X = self.dataX.randn(sample_size=self.batchsize).T

    def rand_draw_X_fixed(self):
        self.X_fixed = np.squeeze(self.dataX.randn(sample_size=1))

    def rand_draw_X_fixed_with_index(self, xi):
        X_fixed = np.squeeze(self.dataX.randn(sample_size=1))
        self.X_fixed[xi] = X_fixed[xi]
        if len(self.fixed_column):
            self.X[:, self.fixed_column] = self.X_fixed[self.fixed_column]

    def rand_draw_data_with_X_fixed(self):
        self.X = self.dataX.randn(sample_size=self.batchsize).T
        if len(self.fixed_column):
            self.X[:, self.fixed_column] = self.X_fixed[self.fixed_column]

    def evaluate(self):
        return self.data_query_oracle.evaluate(self.X)

    def reward_function(self, p):
        y_hat = p.execute(self.X)
        return self.data_query_oracle._evaluate_loss(self.X, y_hat)
