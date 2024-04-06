import os

import json
from cryptography.fernet import Fernet
from sympy import Symbol
from sympy import parse_expr

import time
from scibench.metrics import all_metrics, tree_edit_distance
from scibench.tokens import *
from scibench.program import *

EQUATION_EXTENSION = ".in"




class Equation_evaluator(object):
    def __init__(self, true_equation_filename, noise_type='normal', noise_scale=0.0, metric_name="neg_nmse"):
        '''
        true_equation_filename: the program to map from X to Y
        noise_type, noise_scale: the type and scale of noise.
        metric_name: evaluation metric name for `y_true` and `y_pred`
        '''

        self.true_equation, self.num_vars, self.dim, self.operators_set, self.vars_range_and_types, self.expr = self.__load_equation(
            true_equation_filename)
        # metric
        self.metric_name = metric_name
        self.metric = all_metrics[metric_name]

        # noise
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.noises = construct_noise(self.noise_type)
        self.start = time.time()

    # Declaring private method. This function cannot be called outside the class.
    def __load_equation(self, equation_name, key_filename="encrypted_equation/public.key"):
        """
        load the equation. For encrypted equation extra key file is needed.
        """
        self.eq_name = equation_name
        if not os.path.isfile(self.eq_name):
            raise FileNotFoundError(f"{self.eq_name} not found!")

        one_equation = decrypt_equation(self.eq_name, key_filename=key_filename)
        num_vars = int(one_equation['num_vars'])
        kwargs_list = [{'real': True} for _ in range(num_vars)]

        assert len(kwargs_list) == num_vars
        self.num_vars = num_vars

        x = [Symbol(f'X{i}', **kwargs) for i, kwargs in enumerate(kwargs_list)]
        return one_equation['eq_expression'], int(one_equation['num_vars']), one_equation['dim'], one_equation['function_set'], \
            one_equation['vars_range_and_types'], parse_expr(one_equation['expr'])

    def evaluate(self, X, debug_mode=False):
        """
        evaluate the y_true from given input X
        """
        batch_size, nvar = X.shape
        assert self.num_vars == nvar, f"The number of variables in your input is {nvar}, but we expect {self.num_vars}"

        if self.true_equation is None:
            raise NotImplementedError('no equation is available')
        y_true = self.true_equation.execute(X) + self.noises(self.noise_scale, batch_size)
        """
        the following part is used to double check if the preorder traversal correctly computes the output.
        """
        if debug_mode:
            y_hat = self.get_symbolic_output(X) + self.noises(self.noise_scale, batch_size)
            for y_i, y_hat_i in zip(y_true, y_hat):
                if np.abs(y_i - y_hat_i) > 1e-10:
                    raise ArithmeticError(f'the difference are too large {y_i} {y_hat_i}')
        return y_true

    def execute_simulate(self, c0: np.ndarray, simulate_steps=20000, return_last_step=True):
        all_c = [c0]
        c = c0
        for i in range(simulate_steps):
            c_new = self.true_equation.execute(c)
            if return_last_step:
                all_c = c_new
            else:
                all_c.append(c_new)
            c = c_new
        return all_c

    def get_symbolic_output(self, X_test):
        var_x = self.expr.free_symbols
        y_hat = np.zeros(X_test.shape[0])
        for idx in range(X_test.shape[0]):
            X = X_test[idx, :]
            val_dict = {}
            for x in var_x:
                i = int(x.name[2:])
                val_dict[x] = X[i]
            y_hat[idx] = self.expr.evalf(subs=val_dict)
        return y_hat

    def _evaluate_loss(self, X, y_pred):
        """
        Compute the y_true based on the input X. And then evaluate the metric value between y_true and y_pred
        """
        y_true = self.evaluate(X)
        if self.metric_name in ['neg_nmse', 'neg_nrmse', 'inv_nrmse', 'inv_nmse']:
            loss_val = self.metric(y_true, y_pred, np.var(y_true))
        elif self.metric_name in ['neg_mse', 'neg_rmse', 'inv_mse']:
            loss_val = self.metric(y_true, y_pred)
        return loss_val

    def compute_metric(self, y_pred, y_true, var_y_true):
        """
        Compute the y_true based on the input X. And then evaluate the metric value between y_true and y_pred
        """
        if self.metric_name in ['neg_nmse', 'neg_nrmse', 'inv_nrmse', 'inv_nmse']:
            loss_val = self.metric(y_true, y_pred, var_y_true)
        elif self.metric_name in ['neg_mse', 'neg_rmse', 'inv_mse']:
            loss_val = self.metric(y_true, y_pred)
        return loss_val

    def compute_normalized_tree_edit_distance(self, pred_eq):
        return tree_edit_distance(pred_eq, self.expr)

    def _evaluate_all_losses(self, X, y_pred):
        """
        Compute the y_true based on the input X. And then evaluate the metric value between y_true and y_hat.
        Return a dictionary of all the loss values.
        """
        y_true = self.evaluate(X)
        loss_val_dict = {}
        for metric_name in ['neg_nmse', 'neg_nrmse', 'inv_nrmse', 'inv_nmse']:
            metric = all_metrics[metric_name]
            loss_val = metric(y_true, y_pred, np.var(y_true))
            loss_val_dict[metric_name] = loss_val
        for metric_name in ['neg_mse', 'neg_rmse', 'inv_mse']:
            metric = all_metrics[metric_name]
            loss_val = metric(y_true, y_pred)
            loss_val_dict[metric_name] = loss_val

        return loss_val_dict

    def _evaluate_simulate_loss(self, X, y_pred, simulate_steps):
        """
        Compute the y_true based on the input X. And then evaluate the metric value between y_true and y_pred
        """
        y_pred = y_pred.flatten()
        y_true = self.execute_simulate(X, simulate_steps).flatten()

        if self.metric_name in ['neg_nmse', 'neg_nrmse', 'inv_nrmse', 'inv_nmse']:
            loss_val = self.metric(y_true, y_pred, np.var(y_true))
        elif self.metric_name in ['neg_mse', 'neg_rmse', 'inv_mse']:
            loss_val = self.metric(y_true, y_pred)
        else:
            raise NotImplementedError(self.metric_name, "is not implemented....")
        return loss_val

    def _evaluate_simulate_all_losses(self, X, y_pred, simulate_steps):
        """
        Compute the y_true based on the input X. And then evaluate the metric value between y_true and y_pred
        """
        y_pred = y_pred.flatten()
        y_true = self.execute_simulate(X, simulate_steps).flatten()
        loss_val_dict = {}
        for metric_name in ['neg_nmse', 'neg_nrmse', 'inv_nrmse', 'inv_nmse']:
            metric = make_regression_metric(metric_name)
            loss_val = metric(y_true, y_pred, np.var(y_true))
            loss_val_dict[metric_name] = loss_val
        for metric_name in ['neg_mse', 'neg_rmse', 'inv_mse']:
            metric = make_regression_metric(metric_name)
            loss_val = metric(y_true, y_pred)
            loss_val_dict[metric_name] = loss_val
        return loss_val_dict

    def _get_eq_name(self):
        return self.eq_name

    def get_vars_range_and_types(self):
        return self.vars_range_and_types

    def get_nvars(self):
        return self.num_vars

    def get_operators_set(self):
        return self.operators_set


def construct_noise(noise_type):
    _all_samplers = {
        'normal': lambda scale, batch_size: np.random.normal(loc=0.0, scale=scale, size=batch_size),
        'exponential': lambda scale, batch_size: np.random.exponential(scale=scale, size=batch_size),
        'uniform': lambda scale, batch_size: np.random.uniform(low=-np.abs(scale), high=np.abs(scale), size=batch_size),
        'laplace': lambda scale, batch_size: np.random.laplace(loc=0.0, scale=scale, size=batch_size),
        'logistic': lambda scale, batch_size: np.random.logistic(loc=0.0, scale=scale, size=batch_size)
    }
    assert noise_type in _all_samplers, "Unrecognized noise_type" + noise_type

    return _all_samplers[noise_type]


def decrypt_equation(eq_file, key_filename=None):
    with open(eq_file, 'rb') as enc_file:
        encrypted = enc_file.readline()
        if encrypted == b'1\n':
            encrypted = enc_file.readline()
            fernet = Fernet(open(key_filename, 'rb').read())
            decrypted = fernet.decrypt(encrypted)
        elif encrypted == b'0\n':
            decrypted = enc_file.readline()
    one_equation = json.loads(decrypted)
    preorder_traversal = eval(one_equation['eq_expression'])

    preorder_traversal = [tt[0] for tt in preorder_traversal]
    # print(preorder_traversal)
    list_of_tokens = create_tokens(one_equation['num_vars'], one_equation['function_set'], protected=True)
    if 'pow' in preorder_traversal:
        list_of_tokens = list_of_tokens + [sciToken(np.power, "pow", arity=2, complexity=1)]
    protected_library = sciLibrary(list_of_tokens)

    sciProgram.library = protected_library
    sciProgram.set_execute(protected=True, simulated_exec=one_equation['simulated_exec'])
    #
    true_pr = build_program(preorder_traversal, protected_library)
    one_equation['eq_expression'] = true_pr

    print("-" * 20)
    for key in one_equation:
        print(key, "\t", one_equation[key])
    print("-" * 20)
    return one_equation


def build_program(preorder_traversal, library):
    preorder_actions = library.actionize(['const' if is_float(tok) else tok for tok in preorder_traversal])
    true_pr = sciProgram(preorder_actions)
    for loc, tok in enumerate(preorder_traversal):
        if is_float(tok):
            true_pr.traversal[loc] = PlaceholderConstant(tok)
    return true_pr
