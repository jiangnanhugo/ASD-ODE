import copy
import numpy as np
from sympy import Symbol
from sympy.parsing.sympy_parser import parse_expr

from grammar.grammar_program import execute, SymbolicExpression, optimize
from grammar.grammar_utils import pretty_print_expr, expression_to_template, nth_repl


class VectorizedContextFreeGrammar(object):
    # will link to regression_task
    task = None
    # will link to grammarProgram
    program = None
    # threshold for deciding constants as summary or standalone constant
    opt_num_expriments = 5  # number of experiments done for multi-trail control variable experiments
    expr_obj_thres = 1e-6
    expr_consts_thres = 1e-3

    noise_std = 0.0

    """
       A Task in which the search space is a binary tree. Observations include
       the previous action, the parent, the sibling, and/or the number of dangling
       (unselected) nodes.
    """

    OBS_DIM = 4  # action, parent, sibling, dangling

    def __init__(self, nvars, production_rules, start_symbols, non_terminal_nodes,
                 max_length,
                 hof_size, reward_threhold):
        # number of input variables
        self.nvars = nvars
        # input variable symbols
        self.input_var_Xs = [Symbol('X' + str(i)) for i in range(self.nvars)]
        self.production_rules = production_rules

        self.start_symbol = "|".join(["f" for _ in range(self.nvars)])+'->' + start_symbols
        self.non_terminal_nodes = non_terminal_nodes
        self.max_length = max_length
        self.hof_size = hof_size
        self.reward_threhold = reward_threhold
        self.hall_of_fame = []
        self.allowed_grammar = np.ones(len(self.production_rules), dtype=bool)
        # those rules has terminal symbol on the right-hand side
        self.terminal_rules = [g for g in self.production_rules if
                               sum([nt in g[3:] for nt in self.non_terminal_nodes]) == 0]
        self.print_grammar_vocabulary()
        print(f"rules with only terminal symbols: {self.terminal_rules}")

        # used for output vocabulary
        self.n_action_inputs = self.output_vocab_size + 1  # Library tokens + empty token
        self.n_parent_inputs = self.output_vocab_size + 1  # - len(self.terminal_rules)  # Parent sub-lib tokens + empty token
        self.n_sibling_inputs = self.output_vocab_size + 1  # Library tokens + empty token
        self.EMPTY_ACTION = self.n_action_inputs - 1
        self.EMPTY_PARENT = self.n_parent_inputs - 1
        self.EMPTY_SIBLING = self.n_sibling_inputs - 1

    def allowed_grammar_indices(self, vc: set) -> list:
        """
        return the list of indices for all grammars that do not have the rules for controlled variables.
        :param vc: the set of controlled variables
        """
        filtered_grammars = []
        for idx, g in enumerate(self.production_rules):
            if sum([vi in g for vi in vc]) == 0:
                filtered_grammars.append(idx)
        return filtered_grammars

    @property
    def output_vocab_size(self):
        return len(self.production_rules)

    def print_grammar_vocabulary(self):
        print('============== GRAMMAR Vocabulary ==============')
        print('{0: >8} {1: >20}'.format('ID', 'NAME'))
        for i in range(len(self.production_rules)):
            print('{0: >8} {1: >20}'.format(i + 1, self.production_rules[i]))
        print('========== END OF GRAMMAR Vocabulary ===========')

    def valid_production_rules(self, Node):
        # Get index of all possible production rules starting with a given node
        return [self.production_rules.index(x) for x in self.production_rules if x.startswith(Node)]

    def get_non_terminal_nodes(self, prod) -> list:

        # Get all the non-terminal nodes from right-hand side of a production rule grammar
        return [i for i in prod[3:] if i in self.non_terminal_nodes]

    def complete_rules(self, list_of_rules):
        """
        complete all non-terminal symbols in rules.

        given one sequence of rules, either cut the sequence for the position where Number_of_Non_Terminal_Symbols=0,
        or add several rules with non only terminal symbols
        """
        ntn_counts = 0
        for one_rule in list_of_rules:
            ntn_counts += len(self.get_non_terminal_nodes(one_rule)) - 1
            if ntn_counts == 0:
                return list_of_rules
        # print(f"trying to complete all non-terminal in {list_of_rules} ==>", end="\t")
        #
        # for _ in range(ntn_counts):
        #     list_of_rules.append(np.random.choice(self.terminal_rules))
        # print(list_of_rules)
        return list_of_rules

    def construct_expression(self, many_seq_of_rules):
        filtered_many_rules = []
        for one_seq_of_rules in many_seq_of_rules:
            one_seq_of_rules = [self.start_symbol] + [self.production_rules[li] for li in one_seq_of_rules]

            one_list_of_rules = self.complete_rules(one_seq_of_rules)
            filtered_many_rules.append(one_list_of_rules)
            # print("pruned list_of_rules:", one_list_of_rules)
        self.task.rand_draw_data_with_X_fixed()
        y_true = self.task.evaluate()
        if self.program.n_cores == 1:
            many_expressions = self.program.fitting_new_expressions(filtered_many_rules, self.task.X, y_true,
                                                                    self.input_var_Xs)
        elif self.program.n_cores >= 2:
            many_expressions = self.program.fitting_new_expressions_in_parallel(filtered_many_rules, self.task.X, y_true,
                                                                                self.input_var_Xs)
        # for one_expression in many_expressions:
        #     if one_expression.reward != -np.inf:
        #         one_expression.all_metrics = self.print_reward_function_all_metrics(one_expression.fitted_eq)
        return many_expressions

    def update_hall_of_fame(self, one_fitted_expression: SymbolicExpression):

        if one_fitted_expression.traversal.count(';') <= self.max_length:
            if not self.hall_of_fame:
                self.hall_of_fame = [one_fitted_expression]
            elif one_fitted_expression.traversal not in [x.traversal for x in self.hall_of_fame]:
                if len(self.hall_of_fame) < self.hof_size:
                    self.hall_of_fame.append(one_fitted_expression)
                    # sorting the list in ascending order
                    self.hall_of_fame = sorted(self.hall_of_fame, key=lambda x: x.reward, reverse=False)
                else:
                    if one_fitted_expression.reward > self.hall_of_fame[-1].reward:
                        # sorting the list in ascending order
                        self.hall_of_fame = sorted(self.hall_of_fame[1:] + [one_fitted_expression],
                                                   key=lambda x: x.reward, reverse=False)

    def print_hofs(self, mode: str, verbose=False):
        """
        mode: if global, then we rank on no variable controlled.
        """
        self.task.rand_draw_data_with_X_fixed()
        print(f"PRINT Best Equations (free variables={self.task.fixed_column})")
        print("=" * 20)
        for pr in self.hall_of_fame[:self.hof_size]:
            if verbose:
                print('        ', pr, end="\n")
                if pr.reward != -np.inf:
                    self.print_reward_function_all_metrics(pr.fitted_eq, verbose=verbose)
                else:
                    print("No metrics")
            else:
                print('        ', pr, end="\n")
        print("=" * 20)

    def print_and_sort_global_Qs(self, Q):
        """
        mode: if global, then we rank on no variable controlled.
        """
        self.task.rand_draw_data_with_X_fixed()

        for pr in Q:
            fitness_scores = self.print_reward_function_all_metrics(pr.fitted_eq, verbose=False)
            pr.reward = fitness_scores[self.program.metric_name]
            pr.all_metrics = fitness_scores
        Q = sorted(Q, key=lambda x: x.reward, reverse=False)

        print("=" * 20)
        for pr in Q:
            print('        ', pr, end="\n")
            pr.print_all_metrics()
        print("=" * 20)

        return Q

    def print_reward_function_all_metrics(self, expr_str, verbose=False):
        """used for print the error for all metrics between the predicted program `p` and true program."""
        y_hat = execute(expr_str, self.task.X.T, self.input_var_Xs)
        dict_of_result = self.task.data_query_oracle._evaluate_all_losses(self.task.X, y_hat)
        # dict_of_result['tree_edit_distance'] = self.task.data_query_oracle.compute_normalized_tree_edit_distance(
        #     expr_str)
        if verbose:
            print('-' * 30)
            for mertic_name in dict_of_result:
                print(f"{mertic_name} {dict_of_result[mertic_name]}")
            print('-' * 30)
        return dict_of_result
