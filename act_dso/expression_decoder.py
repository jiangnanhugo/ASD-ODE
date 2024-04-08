import tensorflow as tf
import numpy as np

from grammar.grammar import ContextFreeGrammar
from grammar.memory import Batch
from grammar.subroutines import parents_siblings


class LinearWrapper:
    """
    RNNCell wrapper that adds a linear layer to the output.

    See: https://github.com/tensorflow/models/blob/master/research/brain_coder/single_task/pg_agent.py
    """

    def __init__(self, cell, output_size):
        self.cell = cell
        self._output_size = output_size

    def __call__(self, inputs, state, scope=None):
        with tf.compat.v1.variable_scope(type(self).__name__):
            outputs, state = self.cell(inputs, state, scope=scope)
            logits = tf.compat.v1.layers.dense(outputs, units=self._output_size)

        return logits, state

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self.cell.state_size

    def zero_state(self, batch_size, dtype):
        return self.cell.zero_state(batch_size, dtype)


class NeuralExpressionDecoder(object):
    """
    Recurrent neural network (RNN) used to generate expressions. Specifically, the RNN outputs a distribution over the
    production rules of symbolic expression. It is trained using REINFORCE with baseline.

    sess : tf.Session. TenorFlow Session object.
    input_embedding_layer:  Object that handles the state features to be used
    debug : int. Debug level. 0: No debug. 1: Print shapes and number of parameters for each variable.
    max_length : int.  Maximum sequence length.
    """

    def __init__(self,
                 # grammar
                 cfg: ContextFreeGrammar,
                 sess,
                 input_embedding_layer,
                 # RNN cell hyperparameters
                 cell: str = 'lstm',  # cell : str Recurrent cell to use. Supports 'lstm' and 'gru'.
                 num_layers: int = 1,  # Number of RNN layers.
                 num_units: int = 32,  # Number of RNN cell units in each of the RNN's layers.
                 initializer: str = 'zeros',
                 # Optimizer hyperparameters
                 optimizer: str = 'adam',
                 learning_rate: float = 0.01,
                 # Loss hyperparameters
                 entropy_weight=0.005,  # Coefficient for entropy bonus.
                 entropy_gamma=1.0,  # Gamma in entropy decay.
                 # PQT hyperparameters
                 pqt: bool = False,  # Train with priority queue training (PQT)?
                 pqt_k=10,  # Size of priority queue.
                 pqt_batch_size=1,  # Size of batch to sample (with replacement) from priority queue.
                 pqt_weight=200.0,  # Coefficient for PQT loss function.
                 pqt_use_pg=False,  # Use policy gradient loss when using PQT?
                 # Other hyperparameters
                 debug=0, max_length=30):

        self.sess = sess
        # set max length of decoding
        self.max_length = max_length

        # Hyperparameters
        self.entropy_weight = entropy_weight
        self.pqt = pqt
        self.pqt_k = pqt_k
        self.pqt_batch_size = pqt_batch_size
        self.cfg = cfg
        decoder_output_vocab_size = cfg.output_vocab_size

        # Placeholders, computed after instantiating expressions
        self.batch_size = tf.compat.v1.placeholder(dtype=tf.int32, shape=(), name="batch_size")
        self.baseline = tf.compat.v1.placeholder(dtype=tf.float32, shape=(), name="baseline")

        # Entropy decay vector
        entropy_gamma_decay = np.array([entropy_gamma ** t for t in range(max_length)])

        # Build controller RNN
        with tf.compat.v1.name_scope("expression_decoder"):

            # Create recurrent cell
            if isinstance(num_units, int):
                num_units = [num_units] * num_layers
            initializer = make_initializer(initializer)
            cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [make_cell(cell, n, initializer=initializer) for n in num_units])
            cell = LinearWrapper(cell=cell, output_size=decoder_output_vocab_size)

            initial_obs = self.initial_obs()
            input_embedding_layer.setup_input_embedding(self, cfg.n_parent_inputs, cfg.n_parent_inputs,
                                                        cfg.n_parent_inputs)
            initial_obs = tf.broadcast_to(initial_obs, [self.batch_size, len(initial_obs)])  # (?, obs_dim)

            # Define loop function to be used by tf.nn.raw_rnn
            def loop_fn(time, cell_output, cell_state, loop_state):

                if cell_output is None:  # time == 0
                    finished = tf.zeros(shape=[self.batch_size], dtype=tf.bool)
                    obs = initial_obs
                    next_input = input_embedding_layer.get_tensor_input(obs)
                    next_cell_state = cell.zero_state(batch_size=self.batch_size,
                                                      dtype=tf.float32)  # 2-tuple, each shape (?, num_units)
                    emit_output = None
                    actions_ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True,
                                                clear_after_read=False)  # Read twice
                    obs_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)

                    lengths = tf.ones(shape=[self.batch_size], dtype=tf.int32)
                    next_loop_state = (actions_ta, obs_ta, obs, lengths,  # Unused until implementing variable length
                                       finished)
                else:
                    actions_ta, obs_ta, obs, lengths, finished = loop_state
                    logits = cell_output
                    next_cell_state = cell_state
                    emit_output = logits
                    # sample from the categorical distribution
                    action = tf.random.categorical(logits=logits, num_samples=1, dtype=tf.int32, seed=1)[:, 0]
                    # Write chosen actions
                    next_actions_ta = actions_ta.write(time - 1, action)
                    # Get current action batch
                    actions = tf.transpose(next_actions_ta.stack())  # Shape: (?, time)

                    # Compute obs
                    next_obs = tf.compat.v1.py_func(func=self.get_next_obs,
                                                    inp=[actions, obs],
                                                    Tout=tf.float32)

                    next_obs.set_shape([None, self.cfg.OBS_DIM])
                    next_input = input_embedding_layer.get_tensor_input(next_obs)
                    next_obs_ta = obs_ta.write(time - 1, obs)  # Write OLD obs
                    finished = next_finished = tf.logical_or(
                        finished,
                        time >= max_length)
                    next_lengths = tf.compat.v1.where(
                        finished,  # Ever finished
                        lengths,
                        tf.tile(tf.expand_dims(time + 1, 0), [self.batch_size]))
                    next_loop_state = (next_actions_ta,
                                       next_obs_ta,
                                       next_obs,
                                       next_lengths,
                                       next_finished)

                return finished, next_input, next_cell_state, emit_output, next_loop_state

            # Returns RNN emit outputs TensorArray (i.e. logits), final cell state, and final loop state
            with tf.compat.v1.variable_scope('policy'):
                _, _, loop_state = tf.compat.v1.nn.raw_rnn(cell=cell, loop_fn=loop_fn)
                actions_ta, obs_ta, _, _, _ = loop_state

            self.actions = tf.transpose(actions_ta.stack(), perm=[1, 0])  # (?, max_length)
            self.obs = tf.transpose(obs_ta.stack(), perm=[1, 2, 0])  # (?, obs_dim, max_length)

        # Generates dictionary containing placeholders needed for a batch of sequences
        def make_batch_ph(name):
            with tf.compat.v1.name_scope(name):
                batch_ph = {
                    "actions": tf.compat.v1.placeholder(tf.int32, [None, max_length]),
                    "obs": tf.compat.v1.placeholder(tf.float32, [None, cfg.OBS_DIM, self.max_length]),
                    "lengths": tf.compat.v1.placeholder(tf.int32, [None, ]),
                    "rewards": tf.compat.v1.placeholder(tf.float32, [None], name="r"),
                }
                batch_ph = Batch(**batch_ph)

            return batch_ph

        def safe_cross_entropy(p, logq, axis=-1):
            safe_logq = tf.compat.v1.where(tf.equal(p, 0.), tf.ones_like(logq), logq)
            return - tf.reduce_sum(p * safe_logq, axis)

        # Generates tensor for neglogp of a given batch
        def make_neglogp_and_entropy(B):
            with tf.compat.v1.variable_scope('policy', reuse=True):
                logits, _ = tf.compat.v1.nn.dynamic_rnn(cell=cell,
                                                        inputs=input_embedding_layer.get_tensor_input(B.obs),
                                                        sequence_length=B.lengths,
                                                        # Backpropagates only through sequence length
                                                        dtype=tf.float32)

            probs = tf.nn.softmax(logits)
            logprobs = tf.nn.log_softmax(logits)

            # Masking from sequence lengths
            # NOTE: Using this mask for neg_log_p and entropy actually does NOT
            # affect training because gradients are zero outside the lengths.
            # However, the mask makes tensorflow summaries accurate.
            mask = tf.sequence_mask(B.lengths, maxlen=max_length, dtype=tf.float32)

            # Negative log probabilities of sequences
            actions_one_hot = tf.one_hot(B.actions, depth=decoder_output_vocab_size, axis=-1, dtype=tf.float32)
            neglogp_per_step = safe_cross_entropy(actions_one_hot, logprobs, axis=2)  # Sum over action dim

            neglogp = tf.reduce_sum(neglogp_per_step * mask, axis=1)  # Sum over time dim

            # If entropy_gamma = 1, entropy_gamma_decay_mask == mask
            entropy_gamma_decay_mask = entropy_gamma_decay * mask  # ->(batch_size, max_length)
            entropy_per_step = safe_cross_entropy(probs, logprobs,
                                                  axis=2)  # Sum over action dim -> (batch_size, max_length)
            entropy = tf.reduce_sum(entropy_per_step * entropy_gamma_decay_mask,
                                    axis=1)  # Sum over time dim -> (batch_size, )

            return neglogp, entropy

        # On policy batch
        self.sampled_batch_ph = make_batch_ph("sampled_batch")

        # Memory batch
        self.memory_batch_ph = make_batch_ph("memory_batch")
        memory_neglogp, _ = make_neglogp_and_entropy(self.memory_batch_ph)
        self.memory_probs = tf.exp(-memory_neglogp)
        self.memory_logps = -memory_neglogp

        # PQT batch
        pqt_loss = None
        if pqt:
            self.pqt_batch_ph = make_batch_ph("pqt_batch")

        # Setup losses
        with tf.compat.v1.name_scope("losses"):
            neglogp, entropy = make_neglogp_and_entropy(self.sampled_batch_ph)
            r = self.sampled_batch_ph.rewards

            # Entropy loss
            entropy_loss = -self.entropy_weight * tf.reduce_mean(entropy, name="entropy_loss")
            loss = entropy_loss

            if not pqt or (pqt and pqt_use_pg):
                # Baseline is the worst of the current samples r
                pg_loss = tf.reduce_mean((r - self.baseline) * neglogp, name="pg_loss")
                # Loss already is set to entropy loss
                loss += pg_loss

            # Priority queue training loss
            if pqt:
                pqt_neglogp, _ = make_neglogp_and_entropy(self.pqt_batch_ph)
                pqt_loss = pqt_weight * tf.reduce_mean(pqt_neglogp, name="pqt_loss")
                loss += pqt_loss

            self.loss = loss

        # compute gradient
        optimizer = make_optimizer(name=optimizer, learning_rate=learning_rate)
        with tf.compat.v1.name_scope("train"):
            self.grads_and_vars = optimizer.compute_gradients(self.loss)
            self.train_op = optimizer.apply_gradients(self.grads_and_vars)
        with tf.compat.v1.name_scope("grad_norm"):
            self.grads, _ = list(zip(*self.grads_and_vars))
            self.norms = tf.linalg.global_norm(self.grads)

        if debug >= 1:
            total_parameters = 0
            print("parameters lists")
            for variable in tf.compat.v1.trainable_variables():
                shape = variable.get_shape()
                n_parameters = np.prod(shape)
                total_parameters += n_parameters
                print("Variable:    ", variable.name)
                print("  Shape:     ", shape)
                print("  Parameters:", n_parameters)
            print("Total parameters:", total_parameters)

    def initial_obs(self):
        """
        Returns the initial observation: empty action, parent, and sibling, and
        dangling is 1.
        """

        # Order of observations: action, parent, sibling
        initial_obs = np.array([self.cfg.EMPTY_ACTION,
                                self.cfg.EMPTY_PARENT,
                                self.cfg.EMPTY_SIBLING,
                                1],
                               dtype=np.float32)
        print("initial_obs:", initial_obs)
        return initial_obs

    def get_next_obs(self, actions, obs):
        print(f"actions: {actions.shape} obs: {obs.shape}")
        dangling = obs[:, 3]  # Shape of obs: (?, 4)

        action = actions[:, -1]  # Current action
        # Compute parents and siblings
        parent, sibling = parents_siblings(actions,
                                           empty_parent=self.cfg.EMPTY_PARENT,
                                           empty_sibling=self.cfg.EMPTY_SIBLING)

        # Update dangling with (arity - 1) for each element in action
        # print("parent {}, sibling {}".format(parent, sibling))
        next_obs = np.stack([action, parent, sibling, dangling], axis=1)  # (?, 4)
        next_obs = next_obs.astype(np.float32)
        # print("next_obs:", next_obs)
        return next_obs

    def sample(self, batch_size):
        """Sample a #batch_size of expressions"""
        feed_dict = {self.batch_size: batch_size}
        actions, obs = self.sess.run([self.actions, self.obs], feed_dict=feed_dict)

        return actions, obs

    def compute_probs(self, memory_batch, log=False):
        """Compute the probabilities of a Batch."""

        feed_dict = {
            self.memory_batch_ph: memory_batch
        }

        if log:
            fetch = self.memory_logps
        else:
            fetch = self.memory_probs
        probs = self.sess.run([fetch], feed_dict=feed_dict)[0]
        return probs

    def train_step(self, b, sampled_batch, pqt_batch):
        """Computes loss and trains model."""
        feed_dict = {
            self.baseline: b,
            self.sampled_batch_ph: sampled_batch
        }

        if self.pqt:
            feed_dict.update({
                self.pqt_batch_ph: pqt_batch
            })

        _ = self.sess.run(self.train_op, feed_dict=feed_dict)


def make_initializer(name):
    "initiailizer : str. Initializer for the recurrent cell. Supports 'zeros' and 'var_scale'."
    if name == "zeros":
        return tf.compat.v1.zeros_initializer()
    if name == "var_scale":
        return tf.compat.v1.keras.initializers.VarianceScaling(
            scale=0.5, mode='fan_avg', distribution=("uniform" if True else "truncated_normal"), seed=0)
    raise ValueError("Did not recognize initializer '{}'".format(name))


def make_cell(name, num_units, initializer):
    if name == 'lstm':
        return tf.compat.v1.nn.rnn_cell.LSTMCell(num_units, initializer=initializer)
    if name == 'gru':
        return tf.compat.v1.nn.rnn_cell.GRUCell(num_units, kernel_initializer=initializer, bias_initializer=initializer)
    raise ValueError("Did not recognize cell type '{}'".format(name))


def make_optimizer(name, learning_rate):
    """optimizer : str. Optimizer to use. ['adam', 'rmsprop', 'sgd']
    learning_rate : float. Learning rate for optimizer.
    """
    if name == "adam":
        return tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    elif name == "rmsprop":
        return tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.99)
    elif name == "sgd":
        return tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
    else:
        raise ValueError("Did not recognize optimizer '{}'".format(name))
