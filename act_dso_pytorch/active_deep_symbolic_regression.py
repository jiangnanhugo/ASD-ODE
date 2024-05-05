"""Core deep symbolic optimizer construct."""

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

from collections import defaultdict
import random
import time

import numpy as np

import torch
from expression_decoder import NeuralExpressionDecoder
from train import learn
from utils import load_config
import sys


class ActDeepSymbolicRegression(object):
    """
    Active Deep symbolic optimization for ODE.
    Includes model hyperparameters and training configuration.
    """

    def __init__(self, config, defined_grammar):
        """config : dict or str. Config dictionary or path to JSON.
        cfg: context-sensitive-grammar
        """
        # set config
        self.config = load_config(config)
        self.config = defaultdict(dict, config)
        self.defined_grammar = defined_grammar
        self.config_training = self.config["training"]
        self.config_expression_decoder = self.config["expression_decoder"]

    def setup(self, device):
        # Generate objects needed for training and set seeds
        # set seeds
        seed = int(time.perf_counter() * 10000) % 1000007
        random.seed(seed)
        print('random seed=', seed)
        seed = int(time.perf_counter() * 10000) % 1000007
        np.random.seed(seed)
        seed = int(time.perf_counter() * 10000) % 1000007
        torch.seed(seed)

        # Prepare training parameters
        max_length = 15
        type = 'lstm'
        num_layers = 2
        hidden_size = 250
        dropout = 0.0
        lr = 0.0005

        self.expression_decoder = NeuralExpressionDecoder(hidden_size,
                                          max_length=max_length, cell=type, dropout=dropout, device=device).to(device)
        if self.config_expression_decoder['optimizer'] == 'adam':
            optim = torch.optim.Adam(self.expression_decoder.parameters(), lr=lr)
        else:
            optim = torch.optim.RMSprop(self.expression_decoder.parameters(), lr=lr)
        # Perform the regression task

        # self.expression_decoder = NeuralExpressionDecoder(self.defined_grammar,
        #                                                   **self.config_expression_decoder)
        self.optim = optim

    def train(self, reward_threshold, n_epochs):
        """
        return the best predicted expression under the current controlled variable settings.
        """
        print("extra arguments:\n {}".format(self.config_training))
        sys.stdout.flush()
        learn(self.defined_grammar,
              self.expression_decoder,
              reward_threshold=reward_threshold,
              n_epochs=n_epochs,
              **self.config_training)
        #
        results = learn(
            self.defined_grammar,
            self.expression_decoder,
            self.optim,
            inner_optimizer='rmsprop',
            inner_lr=0.1,
            inner_num_epochs=25,
            entropy_coefficient=0.005,
            risk_factor=0.95,
            initial_batch_size=2000,
            scale_initial_risk=True,
            batch_size=500,
            n_epochs=500,
            live_print=True,
            summary_print=True
        )
        return results
