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

    def __init__(self, config, ):
        """config : dict or str. Config dictionary or path to JSON.
        cfg: context-sensitive-grammar
        """
        # set config
        self.config = load_config(config)
        self.config = defaultdict(dict, config)
        self.config_training = self.config["training"]
        self.config_expression_decoder = self.config["expression_decoder"]

    def setup(self, defined_grammar):
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

        self.expression_decoder = NeuralExpressionDecoder(defined_grammar,
                                                          **self.config_expression_decoder)

    def train(self, reward_threshold, n_epochs):
        """
        return the best predicted expression under the current controlled variable settings.
        """
        print("extra arguments:\n {}".format(self.config_training))
        sys.stdout.flush()
        learn(self.defined_grammar,
              self.sess,
              self.expression_decoder,
              reward_threshold=reward_threshold,
              n_epochs=n_epochs,
              **self.config_training)
