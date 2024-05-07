"""Defines main training loop for deep symbolic optimization."""

import os

from itertools import compress

import tensorflow as tf
import numpy as np

from grammar.grammar import ContextFreeGrammar
from memory import Batch
import sys

# Ignore TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

"""
    sess : tf.Session. TensorFlow Session object.
    expression_decoder : ExpressionDecoder. ExpressionDecoder object used to generate Programs.

    n_epochs : int or None, optional. Number of epochs to train when n_samples is None.

    n_samples : int or None, optional
        Total number of expressions to sample when n_epochs is None. In this
        case, n_epochs = int(n_samples / batch_size).
    batch_size : int, optional. Number of sampled expressions per epoch.
    complexity : str, optional. Complexity function name, used computing Pareto front.
    alpha : float, optional.  Coefficient of exponentially-weighted moving average of baseline.
    epsilon : float or None, optional.  Fraction of top expressions used for training. None (or equivalently, 1.0) turns off risk-seeking.


    verbose : bool, optional. Whether to print progress.
    save_summary : bool, optional. Whether to write TensorFlow summaries.
    save_all_epoch : bool, optional. Whether to save all rewards for each iteration.

    baseline : str, optional
        Type of baseline to use: grad J = (R - b) * grad-log-prob(expression).
        Choices:
        (1) "ewma_R" : b = EWMA(<R>)
        (2) "R_e" : b = R_e
        (3) "ewma_R_e" : b = EWMA(R_e)
        (4) "combined" : b = R_e + EWMA(<R> - R_e)
        In the above, <R> is the sample average _after_ epsilon sub-sampling and
        R_e is the (1-epsilon)-quantile estimate.

    b_jumpstart : bool, optional
        Whether EWMA part of the baseline starts at the average of the first
        iteration. If False, the EWMA starts at 0.0.

    early_stopping : bool, optional. Whether to stop early if stopping criteria is reached.
    hof : int or None, optional. If not None, number of top Programs to evaluate after training.

    eval_all : bool, optional
        If True, evaluate all Programs. While expensive, this is useful for
        noisy data when you can't be certain of success solely based on reward.
        If False, only the top Program is evaluated each iteration.

    
    debug : int. Debug level, also passed to Controller. 0: No debug. 1: Print initial parameter means. 2: Print parameter means each step.
    use_memory : bool, optional. If True, use memory queue for reward quantile estimation.
    memory_capacity : int. Capacity of memory queue.

    warm_start : int or None
        Number of samples to warm start the memory queue. If None, uses
        batch_size.
  
    save_freq : int or None. Statistics are flushed to file every save_freq epochs (default == 1). If < 0, uses save_freq = inf
  
    Return : dict. A dict describing the best-fit expression (determined by reward).
    """


def learn(grammar_model: ContextFreeGrammar,
          sess, expression_decoder,
          n_epochs=12, batch_size=1000,
          reward_threshold=0.999999,
           epsilon=0.05, verbose=True,

          b_jumpstart=False, early_stopping=True,
          debug=0,
          ):
    """
      Executes the main training loop.
    """

    # Initialize compute graph
    sess.run(tf.compat.v1.global_variables_initializer())

    if debug >= 1:
        print("\nInitial parameter means:")
        print_var_means(sess)

    # Main training loop
    r_best = -np.inf
    prev_r_best = None
    ewma = None if b_jumpstart else 0.0  # EWMA portion of baseline
    nevals = 0  # Total number of sampled expressions (from RL)

    print("-- RUNNING EPOCHS START -------------")
    sys.stdout.flush()
    for epoch in range(n_epochs):
        # Set of str representations for all Programs ever seen

        # Sample batch of expressions from the expression_decoder
        # Shape of actions: (batch_size, max_length)
        # Shape of obs: [(batch_size, max_length)] * 3
        actions, obs = expression_decoder.sample(batch_size)
        # if verbose:
        #     print("sampled actions:", actions)
        grammar_expressions = grammar_model.construct_expression(actions)
        nevals += batch_size
        # Compute rewards (or retrieve cached rewards)
        r = np.array([p.valid_loss for p in grammar_expressions])
        if verbose:
            print("rewards:", r.shape, r)
        r_train = r

        # Need for Vanilla Policy Gradient (epsilon = null)
        p_train = grammar_expressions

        # Update HOF
        for p in grammar_expressions:
            if not p.valid_loss:
                continue
            grammar_model.update_hall_of_fame(p)

        # Store in variables the values for the whole batch (those variables will be modified below)
        r_max = np.max(r)
        r_best = max(r_max, r_best)

        """
        Apply risk-seeking policy gradient: compute the empirical quantile of
        rewards and filter out programs with lesser reward.
        """
        if epsilon is not None and epsilon < 1.0:
            # Compute reward quantile estimate
            quantile = np.nanquantile(r, 1 - epsilon)

            keep = r >= quantile
            print('keep:', np.sum(keep), len(keep), "quantile:", quantile)
            r_train = r = r[keep]
            p_train = grammar_expressions = list(compress(grammar_expressions, keep))

            '''
                get the action, observation status of all programs returned to the controller.
            '''
            actions = actions[keep, :]
            obs = obs[keep, :, :]

        # Clip bounds of rewards to prevent NaNs in gradient descent
        r = np.clip(r, -1e6, 1e6)
        r_train = np.clip(r_train, -1e6, 1e6)
        print("r_train shape:", r_train.shape)
        sys.stdout.flush()
        # Compute baseline
        # NOTE: pg_loss = tf.reduce_mean((self.r - self.baseline) * neglogp, name="pg_loss")
        b_train = quantile

        # Compute sequence lengths
        lengths = np.array([min(len(p.traversal), expression_decoder.max_length) for p in p_train], dtype=np.int32)

        # Create the Batch
        sampled_batch = Batch(actions=actions,
                              obs=obs,
                              lengths=lengths,
                              rewards=r_train)
        expression_decoder.train_step(b_train, sampled_batch)


        # Update new best expression
        new_r_best = False

        if prev_r_best is None or r_max > prev_r_best:
            new_r_best = True
            p_r_best = grammar_expressions[np.argmax(r)]

        prev_r_best = r_best

        # Print new best expression
        if verbose and new_r_best:
            print(" Training epoch {}/{}, current best R: {}".format(epoch + 1, n_epochs, prev_r_best))
            print(f"\t{p_r_best}")

            print("\t** New best")

        if early_stopping and p_r_best.valid_loss > reward_threshold:
            print("Early stopping criteria met; terminate early.")
            return

        if verbose and (epoch + 1) % 10 == 0:
            print("Training epoch {}/{}, current best R: {:.4f}".format(epoch + 1, n_epochs, prev_r_best))

        if debug >= 2:
            print("\nParameter means after epoch {} of {}:".format(epoch + 1, n_epochs))
            print_var_means(sess)

        if verbose and (epoch + 1) == n_epochs:
            print("Ending training after epoch {}/{}, current best R: {:.4f}".format(epoch + 1, n_epochs,
                                                                                     prev_r_best))

        grammar_model.print_hofs(verbose=True)

    sys.stdout.flush()
    return


def print_var_means(sess):
    tvars = tf.compat.v1.trainable_variables()
    tvars_vals = sess.run(tvars)
    for var, val in zip(tvars, tvars_vals):
        print(var.name, "mean:", val.mean(), "var:", val.var())
