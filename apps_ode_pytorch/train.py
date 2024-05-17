# train.py: Contains main training loop (and reward functions) for PyTorch
# implementation of Deep Symbolic Regression.


import numpy as np
import time
import torch
from expression_decoder import NeuralExpressionDecoder
from grammar.grammar import ContextFreeGrammar

###############################################################################
# Main Training loop
###############################################################################
"""Deep Symbolic Regression Training Loop

    ~ Parameters ~
    - optimizer ('adam' or 'rmsprop'): optimizer for expression decoder
    - entropy_coefficient (float): entropy coefficient for expression decoder
    - risk_factor (float, >0, <1): we discard the bottom risk_factor quantile
      when training the expresion decoder
    - sample_batch_size (int): number of sample to be drawn from the expression decoder
    - num_batches (int): number of batches
    - verbose (bool): if true, will print updates during training process

    Returns: A list of four lists:
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
        n_epochs=200,
        entropy_coefficient=0.005,
        risk_factor_epsilon=0.95,
        sample_batch_size=200,
        active_mode='default',
        verbose=True,
):
    epoch_best_rewards = []
    epoch_best_expressions = []

    # Best expression and its performance
    best_expression, best_performance = None, float('-inf')

    # First sampling done outside of loop for initial batch size if desired
    start = time.time()
    sequences, log_probabilities, entropies = expression_decoder.sample_sequence(sample_batch_size)
    log_probabilities = torch.sum(log_probabilities, dim=-1)
    entropies = torch.sum(entropies, dim=-1)
    for i in range(n_epochs):
        # Convert sequences into expressions that can be evaluated
        # Optimize constants of expressions using training data
        grammar_expressions = grammar_model.construct_expression(sequences, active_mode=active_mode)

        # Update the best set of expressions discovered
        for p in grammar_expressions:
            if not p.valid_loss:
                continue
            grammar_model.update_topK_expressions(p)

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
            if verbose:
                print("~ Early Stopping Met ~")
                print(f"""Best Expression: {best_str}""")
            break

        # Compute risk threshold
        quantile = np.nanquantile(rewards, risk_factor_epsilon)
        indices_to_keep = torch.tensor([j for j in range(len(rewards)) if rewards[j] >= quantile])

        if len(indices_to_keep) == 0:
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
        if verbose:
            print(f"""Epoch: {i + 1} ({round(float(time.time() - start), 2)}s elapsed)
            Entropy Loss: {entropy_loss.item()}
            Risk-Seeking Loss: {risk_seeking_loss.item()}
            Total Loss: {loss.item()}
            Best Performance (Overall): {best_performance}
            Best Performance (Epoch): {max(rewards)}
            Best Expression (Overall): {best_expression}
            Best Expression (Epoch): {best_epoch_expression}""")
        # Sample for next batch
        sequences, log_probabilities, entropies = expression_decoder.sample_sequence(sample_batch_size)
        log_probabilities = torch.sum(log_probabilities, dim=-1)
        entropies = torch.sum(entropies, dim=-1)

    print(f"""Time Elapsed: {round(float(time.time() - start), 2)}s
            Epochs Required: {i + 1}
            Best Performance: {round(best_performance.item(), 3)}
            Best Expression: {best_expression}""")

    return epoch_best_rewards, epoch_best_expressions, best_performance, best_expression
