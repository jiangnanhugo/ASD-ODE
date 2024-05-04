# the RNN model used to sample expressions. Supports batched
# sampling of variable length sequences. Can select RNN, LSTM, or GRU models.

import torch.nn as nn
import torch.nn.functional as F
import torch

from grammar.grammar import ContextFreeGrammar
from grammar.memory import Batch
from grammar.subroutines import parents_siblings

class NeuralExpressionDecoder(nn.Module):
    """
    Recurrent neural network (RNN) used to generate expressions. Specifically, the RNN outputs a distribution over the
    production rules of symbolic expression. It is trained using REINFORCE with baseline.
    """

    def __init__(self,
                 # grammar
                 defined_grammar: ContextFreeGrammar,
                 hidden_size, min_length=2, max_length=15,
                 # RNN cell hyperparameters
                 cell: str = 'rnn',  # cell : str Recurrent cell to use. Supports 'lstm' and 'gru'.
                 num_layers: int = 1,  # Number of RNN layers.
                 dropout=0.0, device='cpu'):
        super(NeuralExpressionDecoder, self).__init__()
        self.defined_grammar = defined_grammar
        self.input_vocab_size = defined_grammar.output_rules_size  # One-hot encoded parent and sibling
        self.hidden_size = hidden_size
        self.output_size = defined_grammar.output_rules_size  # Output is a softmax distribution over all operators
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.cell = cell

        # Initial cell optimization
        # self.init_input = nn.Parameter(data=torch.rand(1, self.input_vocab_size), requires_grad=True).to(self.device)
        # self.init_hidden = nn.Parameter(data=torch.rand(self.num_layers, self.hidden_size),
        #                                 requires_grad=True).to(self.device)
        self.embed_layer = nn.Embedding(self.input_vocab_size, hidden_size)
        self.embedding_size = hidden_size

        self.min_length = min_length
        self.max_length = max_length

        if self.cell == 'lstm':
            self.lstm = nn.LSTM(
                input_size=self.input_vocab_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True, proj_size=self.output_size, dropout=self.dropout).to(self.device)
            self.init_hidden_lstm = nn.Parameter(data=torch.rand(self.num_layers, self.output_size),
                                                 requires_grad=True).to(self.device)
        elif self.cell == 'gru':
            self.gru = nn.GRU(
                input_size=self.input_vocab_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout)
            self.projection_layer = nn.Linear(self.hidden_size, self.output_size).to(self.device)
        self.activation = nn.Softmax(dim=1)

    def sample_sequence(self, n,  max_length=15):
        sequences = torch.zeros((n, 0))
        entropies = torch.zeros((n, 0))  # Entropy for each sequence
        log_probs = torch.zeros((n, 0))  # Log probability for each token

        sequence_mask = torch.ones((n, 1), dtype=torch.bool)

        input_tensor = self.init_input.repeat(n, 1)
        hidden_tensor = self.init_hidden.repeat(n, 1)
        if self.cell == 'lstm':
            hidden_lstm = self.init_hidden_lstm.repeat(n, 1)

        counters = torch.ones(n)  # Number of tokens that must be sampled to complete expression
        lengths = torch.zeros(n)  # Number of tokens currently in expression

        # While there are still tokens left for sequences in the batch
        while sequence_mask.all(dim=1).any():
            if self.cell == 'lstm':
                output, hidden_tensor, hidden_lstm = self.forward(input_tensor, hidden_tensor, hidden_lstm)
            elif self.cell == 'gru':
                output, hidden_tensor = self.forward(input_tensor, hidden_tensor)

            # Sample from categorical distribution
            dist = torch.distributions.Categorical(output)
            token = dist.sample()

            # Add sampled tokens to sequences
            sequences = torch.cat((sequences, token[:, None]), axis=1)
            lengths += 1

            # Add log probability of current token
            log_probs = torch.cat((log_probs, dist.log_prob(token)[:, None]), axis=1)

            # Add entropy of current token
            entropies = torch.cat((entropies, dist.entropy()[:, None]), axis=1)

            # Update counter
            counters -= 1
            counters += torch.isin(token, self.operators.arity_two).long() * 2
            counters += torch.isin(token, self.operators.arity_one).long() * 1
            # Update sequence mask
            # This is for the next token that we sample. Basically, we know if the
            # next token will be valid or not based on whether we've just completed the sequence (or have in the past)
            sequence_mask = torch.cat(
                (sequence_mask, torch.bitwise_and((counters > 0)[:, None], sequence_mask.all(dim=1)[:, None])),
                axis=1)

            # Compute next parent and sibling; assemble next input tensor
            parent_sibling = self.get_parent_sibling(sequences, lengths)
            input_tensor = self.get_next_input(parent_sibling)

        # Filter entropies log probabilities using the sequence_mask
        entropies = torch.sum(entropies * (sequence_mask[:, :-1]).long(), axis=1)
        log_probs = torch.sum(log_probs * (sequence_mask[:, :-1]).long(), axis=1)
        sequence_lengths = torch.sum(sequence_mask.long(), axis=1)

        return sequences, sequence_lengths, entropies, log_probs

    def forward(self, input, hidden, hidden_lstm=None):
        """Input should be [parent, sibling]
        """
        if self.cell == 'lstm':
            output, (hn, cn) = self.lstm(input[:, None].float(), (hidden_lstm[None, :], hidden[None, :]))
            output = self.activation(output[:, 0, :])
            return output, cn[0, :], hn[0, :]
        elif self.cell == 'gru':
            output, hn = self.gru(input[:, None].float(), hidden[None, :])
            output = output[:, 0, :]
            output = self.projection_layer(output)
            output = self.activation(output)
            return output, hn[0, :]

    def get_next_input(self, parent_sibling):
        # Just convert -1 to 1 for now; it'll be zeroed out later
        parent = torch.abs(parent_sibling[:, 0]).long()
        sibling = torch.abs(parent_sibling[:, 1]).long()

        # Generate one-hot encoded tensors
        parent_onehot = F.one_hot(parent, num_classes=len(self.operators))
        sibling_onehot = F.one_hot(sibling, num_classes=len(self.operators))

        # Use a mask to zero out values that are -1. Parent should never be -1,
        # but we do it anyway.
        parent_mask = (~(parent_sibling[:, 0] == -1)).long()[:, None]
        parent_onehot = parent_onehot * parent_mask
        sibling_mask = (~(parent_sibling[:, 1] == -1)).long()[:, None]
        sibling_onehot = sibling_onehot * sibling_mask

        input_tensor = torch.cat((parent_onehot, sibling_onehot), axis=1)
        return input_tensor
