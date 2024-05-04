# train.py: Contains main training loop (and reward functions) for PyTorch
# implementation of Deep Symbolic Regression.


import numpy as np
import time
import torch
from torch import nn
from grammar.grammar import ContextFreeGrammar
from grammar.utils import empirical_entropy, weighted_quantile
import math
from grammar.memory import Batch, make_queue
from grammar.variance import quantile_variance
import sys

###############################################################################
# Main Training loop
###############################################################################
"""Deep Symbolic Regression Training Loop

    ~ Parameters ~
    - X_constants (Tensor): X dataset used for training constants
    - y_constants (Tensor): y dataset used for training constants
    - X_rnn (Tensor): X dataset used for obtaining reward / training RNN
    - y_rnn (Tensor): y dataset used for obtaining reward / training RNN
    - operator_list (list of str): operators to use (all variables must have prefix var_)
    - min_length (int): minimum number of operators to allow in expression
    - max_length (int): maximum number of operators to allow in expression
    - type ('rnn', 'lstm', or 'gru'): type of architecture to use
    - num_layers (int): number of layers in RNN architecture
    - dropout (float): dropout (if any) for RNN architecture
    - lr (float): learning rate for RNN
    - optimizer ('adam' or 'rmsprop'): optimizer for RNN
    - inner_optimizer ('lbfgs', 'adam', or 'rmsprop'): optimizer for expressions
    - inner_lr (float): learning rate for constant optimization
    - inner_num_epochs (int): number of epochs for constant optimization
    - entropy_coefficient (float): entropy coefficient for RNN
    - risk_factor (float, >0, <1): we discard the bottom risk_factor quantile
      when training the RNN
    - batch_size (int): batch size for training the RNN
    - num_batches (int): number of batches (will stop early if found)
    - hidden_size (int): hidden dimension size for RNN
    - use_gpu (bool): whether or not to train with GPU
    - live_print (bool): if true, will print updates during training process

    ~ Returns ~
    A list of four lists:
    [0] epoch_best_rewards (list of float): list of highest reward obtained each epoch
    [1] epoch_best_expressions (list of Expression): list of best expression each epoch
    [2] best_reward (float): best reward obtained
    [3] best_expression (Expression): best expression obtained
    """


def learn(
        grammar_model: ContextFreeGrammar,
        expression_decoder,
        operators,
        optim,
        inner_optimizer='rmsprop',
        lr=0.001,
        inner_lr=0.1,
        inner_num_epochs=15,
        entropy_coefficient=0.005,
        risk_factor=0.95,
        initial_batch_size=2000,
        scale_initial_risk=True,
        batch_size=500,
        n_epochs=200,
        live_print=True,
        summary_print=True,
        reward_threshold=0.999999,
        device="cpu"
):
    epoch_best_rewards = []
    epoch_best_expressions = []

    # Best expression and its performance
    best_expression, best_performance = None, float('-inf')

    # First sampling done outside of loop for initial batch size if desired
    start = time.time()
    sequences, sequence_lengths, log_probabilities, entropies = expression_decoder.sample_sequence(initial_batch_size)

    for i in range(n_epochs):
        # Convert sequences into Pytorch expressions that can be evaluated
        expressions = []
        for j in range(len(sequences)):
            expressions.append(
                Expression(operators, sequences[j].long().tolist(), sequence_lengths[j].long().tolist()).to(device)
            )

        # Optimize constants of expressions using training data
        optimize_constants(expressions, X_constants, y_constants, inner_lr, inner_num_epochs, inner_optimizer)

        # Benchmark expressions (test dataset)
        rewards = []
        for expression in expressions:
            rewards.append(benchmark(expression, X_rnn, y_rnn))
        rewards = torch.tensor(rewards)

        # Update best expression
        best_epoch_expression = expressions[np.argmax(rewards)]
        epoch_best_expressions.append(best_epoch_expression)
        epoch_best_rewards.append(max(rewards).item())
        if max(rewards) > best_performance:
            best_performance = max(rewards)
            best_expression = best_epoch_expression

        # Early stopping criteria
        if best_performance >= reward_threshold:
            best_str = str(best_expression)
            if (live_print):
                print("~ Early Stopping Met ~")
                print(f"""Best Expression: {best_str}""")
            break

        # Compute risk threshold
        if i == 0 and scale_initial_risk:
            threshold = np.nanquantile(rewards, 1 - (1 - risk_factor) / (initial_batch_size / batch_size))
        else:
            threshold = np.nanquantile(rewards, risk_factor)
        indices_to_keep = torch.tensor([j for j in range(len(rewards)) if rewards[j] > threshold])

        if len(indices_to_keep) == 0 and summary_print:
            print("Threshold removes all expressions. Terminating.")
            break

        # Select corresponding subset of rewards, log_probabilities, and entropies
        rewards = torch.index_select(rewards, 0, indices_to_keep)
        log_probabilities = torch.index_select(log_probabilities, 0, indices_to_keep)
        entropies = torch.index_select(entropies, 0, indices_to_keep)

        # Compute risk seeking and entropy gradient
        risk_seeking_grad = torch.sum((rewards - threshold) * log_probabilities, axis=0)
        entropy_grad = torch.sum(entropies, axis=0)

        # Mean reduction and clip to limit exploding gradients
        risk_seeking_grad = torch.clip(risk_seeking_grad / len(rewards), -1e6, 1e6)
        entropy_grad = entropy_coefficient * torch.clip(entropy_grad / len(rewards), -1e6, 1e6)

        # Compute loss and back-propagate
        loss = -1 * lr * (risk_seeking_grad + entropy_grad)
        loss.backward()
        optim.step()

        # Epoch Summary
        if live_print:
            print(f"""Epoch: {i + 1} ({round(float(time.time() - start), 2)}s elapsed)
            Entropy Loss: {entropy_grad.item()}
            Risk-Seeking Loss: {risk_seeking_grad.item()}
            Total Loss: {loss.item()}
            Best Performance (Overall): {best_performance}
            Best Performance (Epoch): {max(rewards)}
            Best Expression (Overall): {best_expression}
            Best Expression (Epoch): {best_epoch_expression}
            """)
        # Sample for next batch
        sequences, sequence_lengths, log_probabilities, entropies = expression_decoder.sample_sequence(batch_size)

    if summary_print:
        print(f"""
        Time Elapsed: {round(float(time.time() - start), 2)}s
        Epochs Required: {i + 1}
        Best Performance: {round(best_performance.item(), 3)}
        Best Expression: {best_expression}
        """)

    return [epoch_best_rewards, epoch_best_expressions, best_performance, best_expression]


###############################################################################
# Reward function
###############################################################################

def benchmark(expression, X_rnn, y_rnn):
    """Obtain reward for a given expression using the passed X_rnn and y_rnn
        Compute NRMSE between predicted y and actual y
        """
    with torch.no_grad():
        y_pred = expression(X_rnn)
        loss = nn.MSELoss()
        val = torch.sqrt(loss(y_pred, y_rnn))  # Convert to RMSE
        val = torch.std(y_rnn) * val  # Normalize using stdev of targets
        val = min(torch.nan_to_num(val, nan=1e10), torch.tensor(1e10))  # Fix nan and clip
        val = 1 / (1 + val)  # Squash
        return val.item()


