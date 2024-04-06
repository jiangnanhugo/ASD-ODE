from typing import List

from fractions import Fraction
import numpy as np

from scibench.file_util import is_float


class sciToken(object):
    """
    An arbitrary token or "building block" of a Program object.

    """

    def __init__(self, function, name, arity, complexity, input_var=None):
        """
        name : str. Name of token.
        arity : int. Arity (number of arguments) of token.
        complexity : float. Complexity of token.
        function : callable. Function associated with the token; used for executable Programs.
        input_var : int or None. Index of input if this Token is an input variable, otherwise None.
        """
        self.function = function
        self.name = name
        self.arity = arity
        self.complexity = complexity
        self.input_var = input_var

        if input_var is not None:
            assert function is None, "Input variables should not have functions."
            assert arity == 0, "Input variables should have arity zero."

    def __call__(self, *args):
        """Call the Token's function according to input."""
        assert self.function is not None, "Token {} is not callable.".format(self.name)

        return self.function(*args)

    def __repr__(self):
        return self.name


class PlaceholderConstant(sciToken):
    """
    A Token for placeholder constants that will be optimized with respect to
    the reward function. The function simply returns the "value" attribute.

    Parameters
    ----------
    value : float or None
        Current value of the constant, or None if not yet set.
    """

    def __init__(self, value=None):
        if value is not None:
            value = np.atleast_1d(value)
        self.value = value
        super().__init__(function=self.function, name="const", arity=0, complexity=1)

    def function(self):
        assert self.value is not None, "Constant is not callable with value None."
        return self.value

    def __repr__(self):
        if self.value is None:
            return self.name
        return str(self.value[0])


class HardCodedConstant(sciToken):
    """
    A Token with a "value" attribute, whose function returns the value.
    """

    def __init__(self, value=None, name=None):
        """  Value of the constant. """
        assert value is not None, "Constant is not callable with value None. Must provide a floating point number or string of a float."
        assert is_float(value)
        value = np.atleast_1d(np.float32(value))
        self.value = value
        if name is None:
            name = str(self.value[0])
        super().__init__(function=self.function, name=name, arity=0, complexity=1)

    def function(self):
        return self.value


class sciLibrary(object):
    """
    Library of sciTokens. We use a list of sciTokens (instead of set or dict) since
    we so often index by integers given by the Controller.
    """

    def __init__(self, tokens):
        """
        tokens :List of available Tokens in the library.
        names : list of str, Names corresponding to sciTokens in the library.
        arities : list of int. Arities corresponding to sciTokens in the library.
        """

        self.tokens = tokens
        self.L = len(tokens)
        self.names = [t.name for t in tokens]
        self.arities = np.array([t.arity for t in tokens], dtype=np.int32)

    def print_library(self):
        print('============== LIBRARY ==============')
        print('{0: >8} {1: >10} {2: >8}'.format('ID', 'NAME', 'ARITY'))
        for i in range(self.L):
            print('{0: >8} {1: >10} {2: >8}'.format(i, self.names[i], self.arities[i]))

    def __getitem__(self, val):
        """Shortcut to get Token by name or index."""

        if isinstance(val, str):
            try:
                i = self.names.index(val)
            except ValueError:
                raise ModuleNotFoundError("sciToken {} does not exist.".format(val))
        elif isinstance(val, (int, np.integer)):
            i = val
        else:
            raise ModuleNotFoundError("sciLibrary must be indexed by str or int, not {}.".format(type(val)))

        try:
            token = self.tokens[i]
        except IndexError:
            raise ModuleNotFoundError("sciToken index {} does not exist".format(i))
        return token

    def tokenize(self, inputs):
        """Convert inputs to list of Tokens."""
        if isinstance(inputs, str):
            inputs = inputs.split(',')
        elif not isinstance(inputs, list) and not isinstance(inputs, np.ndarray):
            inputs = [inputs]
        tokens = [input_ if isinstance(input_, sciToken) else self[input_] for input_ in inputs]
        return tokens

    def actionize(self, inputs):
        """Convert inputs to array of 'actions', i.e. ints corresponding to Tokens in the Library."""
        tokens = self.tokenize(inputs)
        actions = np.array([self.tokens.index(t) for t in tokens], dtype=np.int32)
        return actions


GAMMA = 0.57721566490153286060651209008240243104215933593992


def logabs(x1):
    """Closure of log for non-positive arguments."""
    return np.log(np.abs(x1))


def expneg(x1):
    return np.exp(-x1)


def n3(x1):
    return np.power(x1, 3)


def n4(x1):
    return np.power(x1, 4)


def n5(x1):
    return np.power(x1, 5)


def sigmoid(x1):
    return 1 / (1 + np.exp(-x1))


def harmonic(x1):
    if all(val.is_integer() for val in x1):
        return np.array([sum(Fraction(1, d) for d in range(1, int(val) + 1)) for val in x1], dtype=np.float32)
    else:
        return GAMMA + np.log(x1) + 0.5 / x1 - 1. / (12 * x1 ** 2) + 1. / (120 * x1 ** 4)


# Annotate unprotected ops
unprotected_ops = [
    # differential operators
    # sciToken(LaplacianOp, "laplacian", arity=1, complexity=4),
    # sciToken(DifferentialOp, "differential", arity=1, complexity=4),
    # sciToken(ClampOp, "clamp", arity=1, complexity=1),
    # Binary operators
    sciToken(np.add, "add", arity=2, complexity=1),
    sciToken(np.subtract, "sub", arity=2, complexity=1),
    sciToken(np.multiply, "mul", arity=2, complexity=1),
    sciToken(np.power, "pow", arity=2, complexity=1),
    sciToken(np.divide, "div", arity=2, complexity=2),

    # Built-in unary operators
    sciToken(np.sin, "sin", arity=1, complexity=3),
    sciToken(np.cos, "cos", arity=1, complexity=3),
    sciToken(np.tan, "tan", arity=1, complexity=4),
    sciToken(np.exp, "exp", arity=1, complexity=4),
    sciToken(np.log, "log", arity=1, complexity=4),
    sciToken(np.sqrt, "sqrt", arity=1, complexity=4),

    sciToken(np.negative, "neg", arity=1, complexity=1),
    sciToken(np.abs, "abs", arity=1, complexity=2),
    sciToken(np.maximum, "max", arity=1, complexity=4),
    sciToken(np.minimum, "min", arity=1, complexity=4),
    sciToken(np.tanh, "tanh", arity=1, complexity=4),
    sciToken(np.reciprocal, "inv", arity=1, complexity=2),

    # Custom unary operators
    sciToken(logabs, "logabs", arity=1, complexity=4),
    sciToken(expneg, "expneg", arity=1, complexity=4),
    sciToken(np.square, "n2", arity=1, complexity=2),
    sciToken(n3, "n3", arity=1, complexity=3),
    sciToken(n4, "n4", arity=1, complexity=3),
    sciToken(n5, "n5", arity=2, complexity=3),
    sciToken(sigmoid, "sigmoid", arity=1, complexity=4),
    sciToken(harmonic, "harmonic", arity=1, complexity=4)
]

"""Define custom protected operators"""


def protected_div(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


def protected_exp(x1):
    with np.errstate(over='ignore'):
        return np.where(x1 < 100, np.exp(x1), 0.0)


def protected_log(x1):
    """Closure of log for non-positive arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def protected_sqrt(x1):
    """Closure of sqrt for negative arguments."""
    return np.sqrt(np.abs(x1))


def protected_inv(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


def protected_expneg(x1):
    with np.errstate(over='ignore'):
        return np.where(x1 > -100, np.exp(-x1), 0.0)


def protected_n2(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 1e6, np.square(x1), 0.0)


def protected_n3(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 1e6, np.power(x1, 3), 0.0)


def protected_n4(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 1e6, np.power(x1, 4), 0.0)


def protected_sigmoid(x1):
    return 1 / (1 + protected_expneg(x1))


# Annotate protected ops
protected_ops = [
    # Protected binary operators
    sciToken(protected_div, "div", arity=2, complexity=2),

    # Protected unary operators
    sciToken(protected_exp, "exp", arity=1, complexity=4),
    sciToken(protected_log, "log", arity=1, complexity=4),
    sciToken(protected_log, "logabs", arity=1, complexity=4),  # Protected logabs is support, but redundant
    sciToken(protected_sqrt, "sqrt", arity=1, complexity=4),
    sciToken(protected_inv, "inv", arity=1, complexity=2),
    sciToken(protected_expneg, "expneg", arity=1, complexity=4),

    sciToken(protected_n2, "n2", arity=1, complexity=2),
    sciToken(protected_n3, "n3", arity=1, complexity=3),
    sciToken(protected_n4, "n4", arity=1, complexity=3),
    sciToken(protected_sigmoid, "sigmoid", arity=1, complexity=4)
]

# Add unprotected ops to function map
function_map = {
    op.name: op for op in unprotected_ops
}

# Add protected ops to function map
function_map.update({
    "protected_{}".format(op.name): op for op in protected_ops
})

TERMINAL_TOKENS = set([op.name for op in function_map.values() if op.arity == 0])
UNARY_TOKENS = set([op.name for op in function_map.values() if op.arity == 1])
BINARY_TOKENS = set([op.name for op in function_map.values() if op.arity == 2])


def create_tokens(n_input_var: int, function_set: List, protected) -> List:
    """
    Helper function to create Tokens.
    n_input_var : int. Number of input variable Tokens.
    function_set : list. Names of registered Tokens, or floats that will create new Tokens.
    protected : bool. Whether to use protected versions of registered Tokens.
    """

    # Create input variable Tokens
    tokens = [sciToken(name="X{}".format(i), arity=0, complexity=0.0, function=None, input_var=i) for i in range(n_input_var)]

    for op in function_set:
        # Registered Token
        if op in function_map:
            # Overwrite available protected operators
            if protected and not op.startswith("protected_"):
                protected_op = "protected_{}".format(op)
                if protected_op in function_map:
                    op = protected_op
            token = function_map[op]
        # Hard-coded floating-point constant
        elif op == 'const':
            token = PlaceholderConstant(1.0)
        else:
            raise ValueError("Operation {} not recognized.".format(op))
        if token not in tokens:
            tokens.append(token)

    return tokens
