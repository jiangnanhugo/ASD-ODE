import numpy as np
from sympy import Symbol
import math
from grammar.grammar_program import SymbolicDifferentialEquations
from grammar.minimize_coefficients import execute


class ContextFreeGrammar(object):
    # will link to regression_task
    task = None
    # will link to grammarProgram
    program = None

    noise_std = 0.0


    OBS_DIM = 4  # action, parent, sibling, dangling

    def __init__(self, nvars,
                 production_rules, start_symbols, non_terminal_nodes,
                 max_length,
                 hof_size, reward_threhold):
        # number of input variables
        self.nvars = nvars
        # input variable symbols
        self.input_var_Xs = [Symbol('X' + str(i)) for i in range(self.nvars)]
        self.production_rules = production_rules

        self.start_symbol = "||".join(["f" for _ in range(self.nvars)]) + '->' + start_symbols
        self.non_terminal_nodes = non_terminal_nodes
        self.max_length = max_length
        self.hof_size = hof_size
        self.reward_threhold = reward_threhold
        self.best_predicted_equations = []
        self.allowed_grammar = np.ones(len(self.production_rules), dtype=bool)
        # those rules has terminal symbol on the right-hand side
        self.terminal_rules = [g for g in self.production_rules if
                               sum([nt in g[3:] for nt in self.non_terminal_nodes]) == 0]
        self.print_grammar_rules()
        print(f"rules with only terminal symbols: {self.terminal_rules}")

        # used for output vocabulary
        self.n_action_inputs = self.output_rules_size + 1  # Library tokens + empty token
        self.n_parent_inputs = self.output_rules_size + 1  # - len(self.terminal_rules)  # Parent sub-lib tokens + empty token
        self.n_sibling_inputs = self.output_rules_size + 1  # Library tokens + empty token
        self.EMPTY_ACTION = self.n_action_inputs - 1
        self.EMPTY_PARENT = self.n_parent_inputs - 1
        self.EMPTY_SIBLING = self.n_sibling_inputs - 1

    @property
    def output_rules_size(self):
        return len(self.production_rules)

    def print_grammar_rules(self):
        print('============== GRAMMAR ==============')
        print('{0: >8} {1: >20}'.format('ID', 'NAME'))
        for i in range(len(self.production_rules)):
            print('{0: >8} {1: >20}'.format(i + 1, self.production_rules[i]))
        print('========== END OF GRAMMAR ===========')

    def compatiable_terminal_rules(self, symbol: str) -> list:
        # Get index of all possible production rules starting with a given node
        return [x for x in self.terminal_rules if x.startswith(symbol)]

    def extract_non_terminal_nodes(self, prod: str):
        """
        right +1; left -1
        """
        cnt = -1
        for g in prod[3:]:
            if g in self.non_terminal_nodes:
                cnt += 1
        return prod[0], cnt

    def complete_rules(self, list_of_rules):
        """
        complete all non-terminal symbols in rules.
        given one sequence of rules, either cut the sequence for the position where Number_of_Non_Terminal_Symbols=0,
        or add several rules with non-terminal symbols
        """
        filtered_rules = [self.start_symbol, ]
        is_done = set()
        ntn_counts = {}
        for nt in self.start_symbol.split('->')[-1]:
            if nt in self.non_terminal_nodes:
                ntn_counts[nt] = 1
        for one_rule in list_of_rules:
            symbol, extracted_cnt = self.extract_non_terminal_nodes(one_rule)
            if symbol not in is_done:
                filtered_rules.append(one_rule)
            ntn_counts[symbol] += extracted_cnt

            if ntn_counts[symbol] == 0:
                is_done.add(symbol)
        if len(is_done) == len(ntn_counts.keys()):
            return filtered_rules
        print(f"trying to complete all non-terminal  ==>", end="\t")
        #
        for k in ntn_counts:
            if ntn_counts[k] <= 0: continue
            compatiable_rules = self.compatiable_terminal_rules(k)
            random_rules = np.random.choice(compatiable_rules, size=ntn_counts[k])
            filtered_rules.extend(random_rules)
        print(filtered_rules)
        return filtered_rules

    def construct_expression(self, many_seq_of_rules, mode='default'):
        """
        mode
        - "default": validate on randomly chosen data
        - "active_region": validate on actively chosen regions
        - "full": validate on all trajecotries
        """
        filtered_many_rules = []
        for one_seq_of_rules in many_seq_of_rules:
            one_seq_of_rules = [self.production_rules[li] for li in one_seq_of_rules]

            one_list_of_rules = self.complete_rules(one_seq_of_rules)
            filtered_many_rules.append(one_list_of_rules)
            # print("pruned list_of_rules:", one_list_of_rules)
        self.task.rand_draw_init_cond()
        true_trajectories = self.task.evaluate()
        many_expressions = []
        if self.program.n_cores == 1:
            many_expressions = self.program.fitting_new_expressions(
                filtered_many_rules,
                self.task.init_cond, self.task.time_span, self.task.t_evals,
                true_trajectories,
                self.input_var_Xs)
        elif self.program.n_cores > 1:
            many_expressions = self.program.fitting_new_expressions_in_parallel(
                filtered_many_rules,
                self.task.init_cond, self.task.time_span, self.task.t_evals,
                true_trajectories,
                self.input_var_Xs)

        # evaluate the fitted expressions on new validation proc_data;
        if mode =='default':
            self.task.rand_draw_init_cond()
        elif mode =='active_region':
            self.task.active_region_init_cond()
        elif mode == 'full':
            self.task.full_init_cond()

        for one_expression in many_expressions:
            if one_expression.train_loss is not None and one_expression.train_loss != -np.inf:
                # print('\t', one_expression)
                one_ode_str = [str(one_eq) for one_eq in one_expression.fitted_eq]
                pred_trajectories = execute(one_ode_str,
                                            self.task.init_cond, self.task.time_span, self.task.t_evals,
                                            self.input_var_Xs)
                if pred_trajectories is None or len(pred_trajectories) == 0:
                    one_expression.valid_loss = -np.inf
                else:
                    one_expression.valid_loss = self.task.evaluate_loss(pred_trajectories)

            else:
                one_expression.valid_loss = -np.inf
            print(one_expression)
        return many_expressions

    def update_hall_of_fame(self, one_fitted_expression: SymbolicDifferentialEquations):
        # the best equations should be placed at the top 1 slot of self.hall_of_fame
        if one_fitted_expression.traversal.count(';') <= self.max_length:
            if not self.best_predicted_equations:
                self.best_predicted_equations = [one_fitted_expression]
            elif one_fitted_expression.traversal not in [x.traversal for x in self.best_predicted_equations]:
                if len(self.best_predicted_equations) < self.hof_size:
                    self.best_predicted_equations.append(one_fitted_expression)
                    # sorting the list in descending order
                    self.best_predicted_equations = sorted(self.best_predicted_equations,
                                                           key=lambda x: x.train_loss,
                                                           reverse=False)
                else:
                    if one_fitted_expression.train_loss > self.best_predicted_equations[-1].train_loss:
                        # sorting the list in descending order
                        self.best_predicted_equations = sorted(
                            self.best_predicted_equations[1:] + [one_fitted_expression],
                            key=lambda x: x.train_loss,
                            reverse=False)

    def print_hofs(self, verbose=False):
        self.task.rand_draw_init_cond()
        print(f"PRINT Best Equations")
        print("=" * 20)
        for pr in self.best_predicted_equations[:self.hof_size]:
            if verbose:
                print('        ', pr, end="\n")
                # do not print expressions with NaN or Infty value.
                if pr.train_loss != -np.inf and not np.isnan(pr.train_loss) and not np.isnan(pr.valid_loss):

                    pred_trajectories = execute(pr.fitted_eq,
                                                self.task.init_cond, self.task.time_span, self.task.t_evals,
                                                self.input_var_Xs)
                    dict_of_result = self.task.evaluate_all_losses(pred_trajectories)

                    if verbose:
                        print('-' * 30)
                        for metric_name in dict_of_result:
                            print(f"{metric_name} {dict_of_result[metric_name]}")
                        print('-' * 30)
                else:
                    print("No metrics")
            else:
                print('        ', pr, end="\n")
        print("=" * 20)
