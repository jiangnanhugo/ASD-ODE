# train.py: Contains main training loop (and reward functions) for PyTorch
# implementation of Deep Symbolic Regression.


import numpy as np
import time
import torch
from torch import nn
from expression_decoder import NeuralExpressionDecoder
from grammar.grammar import ContextFreeGrammar

###############################################################################
# Main Training loop
###############################################################################
"""Deep Symbolic Regression Training Loop

    ~ Parameters ~
    - operator_list (list of str): operators to use (all variables must have prefix var_)
    - max_length (int): maximum number of operators to allow in expression
    - type ('rnn', 'lstm', or 'gru'): type of architecture to use
    - num_layers (int): number of layers in RNN architecture
    - dropout (float): dropout (if any) for RNN architecture
    - optimizer ('adam' or 'rmsprop'): optimizer for RNN
    - inner_optimizer ('lbfgs', 'adam', or 'rmsprop'): optimizer for expressions
    - inner_num_epochs (int): number of epochs for constant optimization
    - entropy_coefficient (float): entropy coefficient for RNN
    - risk_factor (float, >0, <1): we discard the bottom risk_factor quantile
      when training the RNN
    - batch_size (int): batch size for training the RNN
    - num_batches (int): number of batches (will stop early if found)
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
        expression_decoder: NeuralExpressionDecoder,
        optim,
        reward_threshold=0.999999,
        entropy_coefficient=0.005,
        risk_factor=0.95,
        initial_batch_size=200,
        scale_initial_risk=True,
        batch_size=200,
        n_epochs=200,

        live_print=True,
        summary_print=True,
):
    epoch_best_rewards = []
    epoch_best_expressions = []

    # Best expression and its performance
    best_expression, best_performance = None, float('-inf')

    # First sampling done outside of loop for initial batch size if desired
    start = time.time()
    sequences, sequence_lengths, log_probabilities, entropies = expression_decoder.sample_sequence(batch_size)

    for i in range(n_epochs):
        # Convert sequences into expressions that can be evaluated
        # Optimize constants of expressions using training data
        grammar_expressions = grammar_model.construct_expression(sequences)

        # Update HOF
        for p in grammar_expressions:
            if not p.valid_loss:
                continue
            grammar_model.update_hall_of_fame(p)

        # Benchmark expressions (test dataset)
        # Compute rewards (or retrieve cached rewards)
        rewards = np.array([p.valid_loss for p in grammar_expressions])
        rewards = torch.tensor(rewards)

        # Update best expression
        best_epoch_expression = grammar_expressions[np.argmax(rewards)]
        epoch_best_expressions.append(best_epoch_expression)
        epoch_best_rewards.append(max(rewards).item())
        if max(rewards) > best_performance:
            best_performance = max(rewards)
            best_expression = best_epoch_expression

        # Early stopping criteria
        if best_performance >= reward_threshold:
            best_str = str(best_expression)
            if live_print:
                print("~ Early Stopping Met ~")
                print(f"""Best Expression: {best_str}""")
            break

        # Compute risk threshold
        quantile = np.nanquantile(rewards, risk_factor)
        indices_to_keep = torch.tensor([j for j in range(len(rewards)) if rewards[j] >= quantile])

        if len(indices_to_keep) == 0 and summary_print:
            print("quantile threshold removes all expressions. Terminating.")
            break

        # Select corresponding subset of rewards, log_probabilities, and entropies
        rewards = torch.index_select(rewards, 0, indices_to_keep)
        log_probabilities = torch.index_select(log_probabilities, 0, indices_to_keep)
        entropies = torch.index_select(entropies, 0, indices_to_keep)

        # Compute risk seeking and entropy gradient
        risk_seeking_loss = torch.sum((rewards - quantile) * log_probabilities, axis=0)
        entropy_loss = torch.sum(entropies, axis=0)

        # Mean reduction and clip to limit exploding gradients
        risk_seeking_loss = torch.clip(risk_seeking_loss / len(rewards), -1e6, 1e6)
        entropy_loss = entropy_coefficient * torch.clip(entropy_loss / len(rewards), -1e6, 1e6)

        # Compute loss and back-propagate
        loss = -1 * (risk_seeking_loss + entropy_loss)
        loss.backward()
        optim.step()

        # Epoch Summary
        if live_print:
            print(f"""Epoch: {i + 1} ({round(float(time.time() - start), 2)}s elapsed)
            Entropy Loss: {entropy_loss.item()}
            Risk-Seeking Loss: {risk_seeking_loss.item()}
            Total Loss: {loss.item()}
            Best Performance (Overall): {best_performance}
            Best Performance (Epoch): {max(rewards)}
            Best Expression (Overall): {best_expression}
            Best Expression (Epoch): {best_epoch_expression}""")
        # Sample for next batch
        sequences, sequence_lengths, log_probabilities, entropies = expression_decoder.sample_sequence(batch_size)

    if summary_print:
        print(f"""Time Elapsed: {round(float(time.time() - start), 2)}s
        Epochs Required: {i + 1}
        Best Performance: {round(best_performance.item(), 3)}
        Best Expression: {best_expression}""")

    return epoch_best_rewards, epoch_best_expressions, best_performance, best_expression
