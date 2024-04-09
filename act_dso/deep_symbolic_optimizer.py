"""Core deep symbolic optimizer construct."""

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

from collections import defaultdict
import random
import time

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from expression_decoder import NeuralExpressionDecoder
from train import learn
from utils import load_config
from inputEmbeddingLayer import make_embedding_layer
import sys

class ActDeepSymbolicRegression(object):
    """
    Active Deep symbolic optimization for ODE. Includes model hyperparameters and
    training configuration.
    """

    def __init__(self, config, cfg):
        """config : dict or str. Config dictionary or path to JSON.
        cfg: context-sensitive-grammar
        """
        # set config
        self.set_config(config)
        self.sess = None
        self.cfg = cfg
        self.config_task['batchsize'] = self.config_training['batch_size']

    def setup(self):
        # Clear the cache and reset the compute graph
        tf.compat.v1.reset_default_graph()
        # Generate objects needed for training and set seeds
        # set seeds
        seed = int(time.perf_counter() * 10000) % 1000007
        random.seed(seed)
        print('random seed=', seed)
        seed = int(time.perf_counter() * 10000) % 1000007
        np.random.seed(seed)
        seed = int(time.perf_counter() * 10000) % 1000007
        tf.compat.v1.random.set_random_seed(seed)
        self.sess = tf.compat.v1.Session()

        # Prepare training parameters
        self.input_embedding_layer = make_embedding_layer(self.config_input_embedding)
        self.expression_decoder = NeuralExpressionDecoder(self.cfg,
                                                          self.sess,
                                                          self.input_embedding_layer,
                                                          **self.config_expression_decoder)

    def train(self, reward_threshold, n_epochs):
        """
        return the best predicted expression under the current controlled variable settings.
        """
        print("extra arguments:\n {}".format(self.config_training))
        sys.stdout.flush()
        result_dict = learn(self.cfg,
                            self.sess,
                            self.expression_decoder,
                            reward_threshold=reward_threshold,
                            n_epochs=n_epochs,
                            **self.config_training)

        return result_dict

    def set_config(self, config):
        config = load_config(config)
        self.config = defaultdict(dict, config)
        self.config_task = self.config["task"]
        self.config_training = self.config["training"]
        self.config_input_embedding = self.config["input_embedding"]
        self.config_expression_decoder = self.config["expression_decoder"]

