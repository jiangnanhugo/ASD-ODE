import numpy as np
import array
from scibench.tokens import PlaceholderConstant
from scibench import cyfunc



class sciProgram(object):
    """
    The executable program representing the symbolic expression.

    The program comprises unary/binary operators, input variables, and hard-coded constants.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library.

    Attributes
    ----------
    traversal : list
        List of operators (type: Function) and terminals (type: int, float, or
        str ("const")) encoding the pre-order traversal of the expression tree.

    tokens : np.ndarry (dtype: int)
        Array of integers whose values correspond to indices

    """
    task = None  # Task
    library = None  # Library
    execute = None  # Link to execute. Either cython or python

    def __init__(self, tokens=None):
        """
        Builds the Program from a list of of integers corresponding to Tokens.
        """
        # Can be empty if we are unpickling
        if tokens is not None:
            self._init(tokens)

    def _init(self, tokens):
        # pre-order of the program. the most important thing.
        self.traversal = [sciProgram.library[t] for t in tokens]

        # position of the constant
        self.const_pos = [i for i, t in enumerate(self.traversal) if isinstance(t, PlaceholderConstant)]

        self.len_traversal = len(self.traversal)

        self.invalid = False  # always false.
        self.str = tokens.tostring()
        self.tokens = tokens

    @classmethod
    def set_execute(cls, protected, simulated_exec=False):
        """Sets which execute method to use"""

        if simulated_exec == True:
            execute_function = python_execute2d
        else:
            # execute_function = python_execute

            execute_function = cython_execute

        if protected:
            sciProgram.protected = True
            sciProgram.execute_function = execute_function
        else:
            sciProgram.protected = False

            class InvalidLog():
                """Log class to catch and record numpy warning messages"""

                def __init__(self):
                    self.error_type = None  # One of ['divide', 'overflow', 'underflow', 'invalid']
                    self.error_node = None  # E.g. 'exp', 'log', 'true_divide'
                    self.new_entry = False  # Flag for whether a warning has been encountered during a call to Program.execute()

                def write(self, message):
                    """This is called by numpy when encountering a warning"""

                    if not self.new_entry:  # Only record the first warning encounter
                        message = message.strip().split(' ')
                        self.error_type = message[1]
                        self.error_node = message[-1]
                    self.new_entry = True

                def update(self):
                    """If a floating-point error was encountered, set Program.invalid
                    to True and record the error type and error node."""

                    if self.new_entry:
                        self.new_entry = False
                        return True, self.error_type, self.error_node
                    else:
                        return False, None, None

            invalid_log = InvalidLog()
            np.seterrcall(invalid_log)  # Tells numpy to call InvalidLog.write() when encountering a warning

            # Define closure for execute function
            def unsafe_execute(traversal, X):
                """This is a wrapper for execute_function. If a floating-point error
                would be hit, a warning is logged instead, p.invalid is set to True,
                and the appropriate nan/inf value is returned. It's up to the task's
                reward function to decide how to handle nans/infs."""

                with np.errstate(all='log'):
                    y = execute_function(traversal, X)
                    invalid, error_node, error_type = invalid_log.update()
                    return y, invalid, error_node, error_type

            sciProgram.execute_function = unsafe_execute

    def execute(self, X):
        """
        Execute program on input X.

        Parameters:
        X : np.array. Input to execute the Program over.

        Returns
        result : np.array or list of np.array
            In a single-object Program, returns just an array.
        """

        if not sciProgram.protected:
            # return some weired error.
            result, self.invalid, self.error_node, self.error_type = sciProgram.execute_function(self.traversal, X)
        else:
            result = sciProgram.execute_function(self.traversal, X)
            # always protected. 1/div
        return result

    def print_expression(self):
        print("\tExpression {}: {}".format(0, self.traversal))

    def __repr__(self):
        """Prints the program's traversal"""
        return ','.join([repr(t) for t in self.traversal])


def python_execute2d(traversal, X):
    """
    Executes the program according to X using Python.

    X : array-like, shape = [batch_size, n_features, n_feature], n_features is the number of features.

    Returns
    -------
    y_hats : array-like, shape = [batch_size, n_features, n_feature]
        The result of executing the program on X.
    """

    apply_stack = []

    for node in traversal:
        apply_stack.append([node])

        while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
            token = apply_stack[-1][0]
            terminals = apply_stack[-1][1:]

            if token.input_var is not None:
                intermediate_result = X[:, :]
            else:
                intermediate_result = token(*terminals)
            if len(apply_stack) != 1:
                apply_stack.pop()
                apply_stack[-1].append(intermediate_result)
            else:
                return intermediate_result

    assert False, "Function should never get here!"
    return None


def python_execute(traversal, X):
    """
    Executes the program according to X using Python.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features], where n_samples is the number of samples and n_features is the number of features.

    Returns
    -------
    y_hats : array-like, shape = [n_samples]
        The result of executing the program on X.
    """

    apply_stack = []

    for node in traversal:
        apply_stack.append([node])

        while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
            token = apply_stack[-1][0]
            terminals = apply_stack[-1][1:]

            if token.input_var is not None:
                intermediate_result = X[:, token.input_var]
            else:
                intermediate_result = token(*terminals)
            if len(apply_stack) != 1:
                apply_stack.pop()
                apply_stack[-1].append(intermediate_result)
            else:
                return intermediate_result

    assert False, "Function should never get here!"
    return None


def cython_execute(traversal, X):
    """
    Execute cython function using given traversal over input X.

    Parameters
    ----------

    traversal : list
        A list of nodes representing the traversal over a Program.
    X : np.array
        The input values to execute the traversal over.

    Returns
    -------

    result : float
        The result of executing the traversal.
    """
    if len(traversal) >= 1:
        is_input_var = array.array('i', [t.input_var is not None for t in traversal])
        return cyfunc.execute(X, len(traversal), traversal, is_input_var)
    else:
        return None
