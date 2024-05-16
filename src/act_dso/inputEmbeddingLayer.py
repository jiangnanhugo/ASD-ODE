import tensorflow as tf


def make_embedding_layer(config):
    """
    config : dict. Parameters
    Returns: InputEmbeddingManager.
    """

    if config is None:
        config = {}

    return InputEmbeddingLayer(**config)


class InputEmbeddingLayer(object):
    """
    Class that uses the previous action, parent, sibling, and/or dangling as
    observations.
    """

    def __init__(self, observe_parent=True, observe_sibling=True,
                 observe_action=False, observe_dangling=False, embedding=False,
                 embedding_dim=8):
        """
        Parameters
        ----------
        observe_parent : bool. Observe the parent of the Token being selected?
        observe_sibling : bool. Observe the sibling of the Token being selected?
        observe_action : bool.  Observe the previously selected Token?
        observe_dangling : bool. Observe the number of dangling nodes?
        embedding : bool.  Use dense embeddings for one-hot categorical inputs?
        embedding_dim : int. Size of embeddings for each categorical input if embedding=True.
        """
        self.observe_parent = observe_parent
        self.observe_sibling = observe_sibling
        self.observe_action = observe_action
        self.observe_dangling = observe_dangling

        # Parameter assertions/warnings
        assert self.observe_action + self.observe_parent + self.observe_sibling + self.observe_dangling > 0, \
            "Must include at least one observation."
        # either one-hot or dense vector
        self.embedding = embedding
        self.embedding_size = embedding_dim

        self.n_action_inputs, self.n_parent_inputs, self.n_sibling_inputs = 1, 1, 1

    def setup_input_embedding(self, expression_decoder,n_action_inputs, n_parent_inputs, n_sibling_inputs):
        """
            Function called inside the controller to perform the needed initializations (e.g., if the tf context is needed)
            :param expression_decoder the controller class
        """
        self.expression_decoder = expression_decoder
        self.max_length = expression_decoder.max_length
        # Create embeddings if needed
        if self.embedding:
            initializer = tf.compat.v1.random_uniform_initializer(minval=-1.0, maxval=1.0, seed=0)
            with tf.compat.v1.variable_scope("embeddings", initializer=initializer):
                if self.observe_action:
                    self.action_embeddings = tf.compat.v1.get_variable("action_embeddings",
                                                                       [n_action_inputs, self.embedding_size],
                                                                       trainable=True)
                if self.observe_parent:
                    self.parent_embeddings = tf.compat.v1.get_variable("parent_embeddings",
                                                                       [n_parent_inputs, self.embedding_size],
                                                                       trainable=True)
                if self.observe_sibling:
                    self.sibling_embeddings = tf.compat.v1.get_variable("sibling_embeddings",
                                                                        [n_sibling_inputs, self.embedding_size],
                                                                        trainable=True)

    def get_tensor_input(self, obs):
        observations = []
        action, parent, sibling, dangling = tf.unstack(obs, axis=1)
        print("action {}, parent {}, sibling {}, dangling {}".format(action, parent, sibling, dangling))

        # Cast action, parent, sibling to int for embedding_lookup or one_hot
        action = tf.cast(action, tf.int32)
        parent = tf.cast(parent, tf.int32)
        sibling = tf.cast(sibling, tf.int32)

        # Action, parent, and sibling inputs are either one-hot or embeddings
        if self.observe_action:
            if self.embedding:
                x = tf.nn.embedding_lookup(self.action_embeddings, action)
            else:
                x = tf.one_hot(action, depth=self.n_action_inputs)
            observations.append(x)
        if self.observe_parent:
            if self.embedding:
                x = tf.nn.embedding_lookup(self.parent_embeddings, parent)
            else:
                x = tf.one_hot(parent, depth=self.n_parent_inputs)
            observations.append(x)
        if self.observe_sibling:
            if self.embedding:
                x = tf.nn.embedding_lookup(self.sibling_embeddings, sibling)
            else:
                x = tf.one_hot(sibling, depth=self.n_sibling_inputs)
            observations.append(x)

        # Dangling input is just the value of dangling
        if self.observe_dangling:
            x = tf.expand_dims(dangling, axis=-1)
            observations.append(x)
        # concatenate all the input vectors togethers.
        input_vector = tf.concat(observations, -1)
        return input_vector
