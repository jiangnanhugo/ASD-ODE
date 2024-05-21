# the decoder model used to sample expressions. Supports batched
# sampling of variable length sequences. Can select RNN, LSTM, or GRU models.

import torch.nn as nn
import torch


class NeuralExpressionDecoder(nn.Module):
    """
    Recurrent neural network (RNN) used to generate expressions. Specifically, the RNN outputs a distribution over the
    production rules of symbolic expression. It is trained using REINFORCE with baseline.
    """

    def __init__(self,
                 output_rules_size,
                 # RNN cell hyperparameters
                 cell: str = 'rnn',  # cell : str Recurrent cell to use. Supports 'lstm' and 'gru'.
                 num_layers: int = 1,  # Number of RNN layers.
                 hidden_size: int = 128,  # hidden size of RNN layer
                 max_length: int = 30,  # maximum length of the RNN decoding
                 dropout: float = 0.5,
                 # Loss hyperparameters
                 entropy_weight=0.005,  # Coefficient for entropy bonus.
                 entropy_gamma=1.0,  # Gamma in entropy decay.
                 # Other hyperparameters
                 device='cpu',
                 debug=0):
        """
            - hidden_size (int): hidden dimension size for RNN
        """
        super(NeuralExpressionDecoder, self).__init__()
        # every grammar rules has one embedding, the start symbol also has the last embedding
        self.input_vocab_size = output_rules_size + 1
        self.hidden_size = hidden_size
        # Output is a softmax distribution over all rules (exclude the start symbol
        self.output_size = output_rules_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.cell = cell

        # Entropy decay vector
        self.entropy_weight = entropy_weight
        self.entropy_gamma_decay = torch.tensor([entropy_gamma ** t for t in range(max_length)])

        self.max_length = max_length
        self.embed_layer = nn.Embedding(self.input_vocab_size, hidden_size)
        self.embedding_size = hidden_size

        self.init_hidden = nn.Parameter(data=torch.rand(self.num_layers, self.hidden_size), requires_grad=True).to(
            self.device)

        if self.cell == 'lstm':
            self.lstm = nn.LSTM(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True, proj_size=self.output_size, dropout=self.dropout).to(self.device)
            self.init_hidden_lstm = nn.Parameter(data=torch.rand(self.num_layers, self.output_size),
                                                 requires_grad=True).to(self.device)
        elif self.cell == 'gru':
            self.gru = nn.GRU(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout)
            self.projection_layer = nn.Linear(self.hidden_size, self.output_size).to(self.device)
        self.activation = nn.Softmax(dim=1)

    def sample_sequence(self, seq_batch_size):
        # [batch_size, sequence_length]
        sequences = torch.zeros((seq_batch_size, self.max_length), dtype=torch.int)
        entropies = torch.zeros((seq_batch_size, self.max_length))  # Entropy for each sequence
        log_probabilities = torch.zeros((seq_batch_size, self.max_length))  # Log probability for each token

        input_tensor = torch.tensor([self.input_vocab_size - 1]).repeat(seq_batch_size, 1)
        hidden_tensor = self.init_hidden.repeat(seq_batch_size, 1)
        if self.cell == 'lstm':
            hidden_lstm = self.init_hidden_lstm.repeat(seq_batch_size, 1)

        for ti in range(self.max_length):
            if self.cell == 'lstm':
                output, hidden_tensor, hidden_lstm = self.forward(input_tensor, hidden_tensor, hidden_lstm)
            elif self.cell == 'gru':
                output, hidden_tensor = self.forward(input_tensor, hidden_tensor)

            # Sample from categorical distribution
            dist = torch.distributions.Categorical(output)
            predicted_token = dist.sample()

            # Add sampled tokens to sequences
            sequences[:, ti] = predicted_token

            # Add log probability of current token
            log_probabilities[:, ti] = dist.log_prob(predicted_token)

            # Add entropy of current token
            entropies[:, ti] = dist.entropy()

            input_tensor = predicted_token.reshape(-1, 1)

        entropies = entropies * self.entropy_gamma_decay
        return sequences, entropies, log_probabilities

    def forward(self, input, hidden, hidden_lstm=None):
        """Input should be [parent, sibling]
        """
        embedded_input = self.embed_layer(input)
        if self.cell == 'lstm':
            output, (hn, cn) = self.lstm(embedded_input, (hidden_lstm[None, :], hidden[None, :]))
            output = self.activation(output[:, 0, :])
            return output, cn[0, :], hn[0, :]
        elif self.cell == 'gru':
            # input: batch_size, sequence_length, embedding_dim
            # hidden: (1, batch_size, hidden_size)
            output, hn = self.gru(embedded_input, hidden[None, :])
            output = output[:, 0, :]
            output = self.projection_layer(output)
            output = self.activation(output)
            return output, hn[0, :]
