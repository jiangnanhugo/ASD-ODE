"""Core deep symbolic optimizer construct."""

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

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
        defined_grammar: context-free-grammar for symbolic expression.
        """
        # set config
        self.config = load_config(config)
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
        torch.manual_seed(seed)

        # Prepare training parameters

        self.expression_decoder = NeuralExpressionDecoder(
            output_rules_size=self.defined_grammar.output_rules_size,
            cell=self.config_expression_decoder['cell'],
            num_layers=self.config_expression_decoder['num_layers'],
            hidden_size=self.config_expression_decoder['hidden_size'],
            max_length=self.config_expression_decoder['max_length'],
            dropout=self.config_expression_decoder['dropout'],
            entropy_weight=self.config_expression_decoder['entropy_weight'],
            entropy_gamma=self.config_expression_decoder['entropy_gamma'],
            device=device
        ).to(device)
        if self.config_expression_decoder['optimizer'] == 'adam':
            optim = torch.optim.Adam(self.expression_decoder.parameters(),
                                     lr=self.config_expression_decoder['learning_rate'])
        elif self.config_expression_decoder['optimizer'] == 'RMSprop':
            optim = torch.optim.RMSprop(self.expression_decoder.parameters(),
                                        lr=self.config_expression_decoder['learning_rate'])
        else:
            optim = torch.optim.SGD(self.expression_decoder.parameters(),
                                    lr=self.config_expression_decoder['learning_rate'])
        # Perform the regression task
        self.optim = optim

    def train(self, reward_threshold, n_epochs):
        """
        use policy gradient to train model.
        return the best predicted expression
        """
        print("extra arguments:\n {}".format(self.config_training))
        sys.stdout.flush()

        results = learn(
            grammar_model=self.defined_grammar,
            expression_decoder=self.expression_decoder,
            optim=self.optim,
            reward_threshold=reward_threshold,
            n_epochs=n_epochs,
            risk_factor_epsilon=self.config_training['risk_factor_epsilon'],
            sample_batch_size=self.config_training['sample_batch_size'],
            verbose=self.config_training['verbose']
        )
        return results
