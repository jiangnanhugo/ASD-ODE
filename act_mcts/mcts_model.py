import copy
import sys
import numpy as np
from collections import defaultdict
from sympy import Symbol
from sympy.parsing.sympy_parser import parse_expr
from production_rules import production_rules_to_expr
from program import execute
from utils import pretty_print_expr, expression_to_template, nth_repl


class MCTS(object):
    """
    hall_of_fame: ranked good expressions.
    """
    task = None  # Task
    program = None
    # constants
    opt_num_expr = 1  # number of experiments done for optimization
    expr_obj_thres = 1e-6
    expr_consts_thres = 1e-3

    noise_std = 0.0

    def __init__(self, base_grammars, aug_grammars, non_terminal_nodes, aug_nt_nodes, max_len, max_module, aug_grammars_allowed,
                 exploration_rate=1 / np.sqrt(2), eta=0.999, max_opt_iter=500):
        # number of input variables
        self.nvars = self.task.data_query_oracle.get_nvars()
        self.input_var_Xs = [Symbol('X' + str(i)) for i in range(self.nvars)]
        self.base_grammars = base_grammars
        self.aug_grammars = aug_grammars
        self.grammars = base_grammars + [x for x in aug_grammars if x not in base_grammars]
        self.aug_nt_nodes = aug_nt_nodes
        self.non_terminal_nodes = non_terminal_nodes
        self.max_len = max_len
        self.max_module = max_module
        self.max_aug = aug_grammars_allowed
        self.hall_of_fame = []
        self.exploration_rate = exploration_rate
        self.UCBs = defaultdict(lambda: np.zeros(len(self.grammars)))
        self.QN = defaultdict(lambda: np.zeros(2))
        self.scale = 0
        self.eta = eta
        self.max_opt_iter = max_opt_iter

    def valid_production_rules(self, Node):
        # Get index of all possible production rules starting with a given node
        return [self.grammars.index(x) for x in self.grammars if x.startswith(Node)]

    def valid_non_termianl_production_rules(self, Node):
        # Get index of all possible production rules starting with a given node
        valid_rules=[]
        for i, x in enumerate(self.grammars):
            if x.startswith(Node) and np.sum([y in x[3:] for y in self.non_terminal_nodes]):
                valid_rules.append(i)
        return valid_rules
        # return [self.grammars.index(x) for x in self.grammars if x.startswith(Node) ]

    def get_non_terminal_nodes(self, prod) -> list:
        # Get all the non-terminal nodes from right-hand side of a production rule grammar
        return [i for i in prod[3:] if i in self.non_terminal_nodes]

    def get_unvisited_children(self, state, node) -> list:
        #  Pick an action to to visit the index of all unvisited child.
        valid_actions = self.valid_production_rules(node)
        return [act for act in valid_actions if self.QN[state + ',' + self.grammars[act]][1] == 0]

    def step(self, state, action_idx, ntn):
        """
        state:      all production rules
        action_idx: index of grammar starts from the current Non-terminal Node
        tree:       the current tree
        ntn:        all remaining non-terminal nodes

        This defines one step of Parse Tree traversal
        return tree (next state), remaining non-terminal nodes, reward, and if it is done
        """
        action = self.grammars[action_idx]
        state = state + ',' + action
        ntn = self.get_non_terminal_nodes(action) + ntn

        if not ntn:
            self.task.rand_draw_init_cond()
            y_true = self.task.evaluate()
            expr_template = production_rules_to_expr(state.split(','))
            reward, eq, _, _ = self.program.optimize(expr_template,
                                                     len(state.split(',')),
                                                     self.task.init_cond,
                                                     y_true,
                                                     self.input_var_Xs,
                                                     eta=self.eta,
                                                     max_opt_iter=self.max_opt_iter)

            return state, ntn, reward, True, eq
        else:
            return state, ntn, 0, False, None

    def freeze_equations(self, list_of_grammars, opt_num_expr, stand_alone_constants, next_free_variable):
        # decide summary constants and stand alone constants.
        print("---------Freeze Equation----------")
        freezed_exprs = []
        aug_nt_nodes = []
        new_stand_alone_constants = stand_alone_constants
        # only use the best
        state, _, expr = list_of_grammars[-1]
        optimized_constants = []
        optimized_obj = []
        expr_template = expression_to_template(parse_expr(expr), stand_alone_constants)
        print('expr template is"', expr_template)
        for _ in range(opt_num_expr):
            self.task.rand_draw_X_fixed()
            self.task.rand_draw_init_cond()
            y_true = self.task.evaluate()
            _, eq, opt_consts, opt_obj = self.program.optimize(expr_template,
                                                               len(state.split(',')),
                                                               self.task.init_cond,
                                                               y_true,
                                                               self.input_var_Xs,
                                                               eta=self.eta,
                                                               max_opt_iter=1000)
            ##
            optimized_constants.append(opt_consts)
            optimized_obj.append(opt_obj)
        optimized_constants = np.asarray(optimized_constants)
        optimized_obj = np.asarray(optimized_obj)
        print(optimized_obj)
        num_changing_consts = expr_template.count('C')
        is_summary_constants = np.zeros(num_changing_consts)
        if np.max(optimized_obj) <= self.expr_obj_thres:
            for ci in range(num_changing_consts):
                print("std", np.std(optimized_constants[:, ci]), end="\t")
                if abs(np.mean(optimized_constants[:, ci])) < 1e-5:
                    print(f'c{ci} is a noisy minial constant')
                    is_summary_constants[ci] = 2
                elif np.std(optimized_constants[:, ci]) <= self.expr_consts_thres:
                    print(f'c{ci} {np.mean(optimized_constants[:, ci])} is a stand-alone constant')
                else:
                    print(f'c{ci}  is a summary constant')
                    is_summary_constants[ci] = 1
            ####
            # summary constant vs controlled variable
            ####
            for ci in range(num_changing_consts):
                if is_summary_constants[ci] != 1:
                    continue
                print(expr_template)
                new_expr_template = nth_repl(copy.copy(expr_template), 'C', str(optimized_constants[-1, ci]), ci + 1)
                print(new_expr_template, ci, np.mean(optimized_constants[:, ci]))
                # optimized_constants = []
                optimized_cond_obj = []
                print('expr template is"', new_expr_template)
                for _ in range(opt_num_expr * 3):
                    self.task.rand_draw_X_fixed_with_index(next_free_variable)
                    y_true = self.task.evaluate()
                    _, eq, opt_consts, opt_obj = self.program.optimize(new_expr_template,
                                                                       len(state.split(',')),
                                                                       self.task.init_cond,
                                                                       y_true,
                                                                       self.input_var_Xs,
                                                                       eta=self.eta,
                                                                       max_opt_iter=1000)
                    ##
                    # optimized_constants.append(opt_consts)
                    optimized_cond_obj.append(opt_obj)
                if np.max(optimized_cond_obj) <= self.expr_obj_thres:
                    print(f'summary constant c{ci} will still be a constant in the next round')
                    is_summary_constants[ci] = 3
                else:
                    print(f'summary constant c{ci} will be a summary constant in the next round')

            ####
            cidx = 0
            new_expr_template = 'B->'
            for ti in expr_template:
                if ti == 'C' and is_summary_constants[cidx] == 1:
                    # real summary constant in the next round
                    new_expr_template += '(A)'
                    cidx += 1
                elif ti == "C" and is_summary_constants[cidx] == 0:
                    # standalone constant
                    est_c = np.mean(optimized_constants[:, cidx])
                    if abs(est_c) < 1e-5:
                        est_c = 0.0
                    new_expr_template += str(est_c)
                    if len(new_stand_alone_constants) == 0 or min([abs(est_c - fi) for fi in new_stand_alone_constants]) < 1e-5:
                        new_stand_alone_constants.append(est_c)
                    cidx += 1
                elif ti == 'C' and is_summary_constants[cidx] == 2:
                    # noise values
                    new_expr_template += '0.0'
                    cidx += 1
                elif ti == 'C' and is_summary_constants[cidx] == 3:
                    # is a summary constant but will still be constant in the next round
                    new_expr_template += 'C'
                    cidx += 1
                else:
                    new_expr_template += ti
            freezed_exprs.append(new_expr_template)
            aug_nt_nodes.append(['A', ] * sum([1 for ti in new_expr_template if ti == 'A']))
            return freezed_exprs, aug_nt_nodes, new_stand_alone_constants

        print("No available expression is found....trying to add the current best guessed...")
        state, _, expr = list_of_grammars[-1]
        expr_template = expression_to_template(parse_expr(expr), stand_alone_constants)
        cidx = 0
        new_expr_template = 'B->'
        for ti in expr_template:
            if ti == 'C':
                # summary constant
                new_expr_template += '(A)'
                cidx += 1
            else:
                new_expr_template += ti
        freezed_exprs.append(new_expr_template)
        aug_nt_nodes.append(['A', ] * sum([1 for ti in new_expr_template if ti == 'A']))
        expri, ntnodei = freezed_exprs[0], aug_nt_nodes[0]
        countA = expri.count('(A)')
        # diversify the number of A
        new_freezed_exprs = [expri, ]
        new_aug_nt_nodes = [['A', ] * countA, ]

        if countA >= 3:
            ti = 0
            while ti < 2:
                mask = np.random.randint(2, size=countA)
                while np.sum(mask) == 0 or np.sum(mask) == countA:
                    mask = np.random.randint(2, size=countA)
                countAi = 0
                expri_new = ""
                for i in range(len(expri)):
                    if expri[i] == 'A' and mask[countAi] == 0:
                        expri_new += 'C'
                    else:
                        expri_new += expri[i]
                    countAi += (expri[i] == 'A')
                if expri_new not in new_freezed_exprs:
                    new_freezed_exprs.append(expri_new)
                    new_aug_nt_nodes.append(['A', ] * (np.sum(mask)))
                    ti += 1
        else:
            new_freezed_exprs.append(expri)
            new_aug_nt_nodes.append(ntnodei)
        # only generate at most 3 template for the next round, otherwise it will be too time consuming
        ret_frezze_exprs, ret_aug_nt_nodes=[], []
        for x, y in zip(new_freezed_exprs, new_aug_nt_nodes):
            if x not in ret_frezze_exprs:
                ret_frezze_exprs.append(x)
                ret_aug_nt_nodes.append(y)
        return ret_frezze_exprs, ret_aug_nt_nodes, new_stand_alone_constants

    def rollout(self, num_play, state_initial, ntn_initial):
        """
        Perform `num_play` simulation, get the maximum reward
        """
        best_eq = ''
        reward = -100
        next_state = None
        eq = ''
        best_r = -100
        idx = 0
        while idx < num_play:
            done = False
            state = state_initial
            ntn = ntn_initial

            while not done:
                valid_index = self.valid_production_rules(ntn[0])
                action = np.random.choice(valid_index)
                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn[1:])
                state = next_state
                ntn = ntn_next

                if state.count(',') >= self.max_len:  # tree depth shall be less than max_len
                    break

            if done:
                idx += 1
                if reward > best_r:
                    # save the current expression into hall-of-fame
                    self.update_hall_of_fame(next_state, reward, eq)
                    best_eq = eq
                    best_r = reward

        return best_r, best_eq

    def update_ucb_mcts(self, state, action):
        """
        Get the ucb score for a given child of current node
        Q and N values are stored in QN matrix.
        """
        next_state = state + ',' + action
        Q_child = self.QN[next_state][0]
        N_parent = self.QN[state][1]
        N_child = self.QN[next_state][1]
        return Q_child / N_child + self.exploration_rate * np.sqrt(np.log(N_parent) / N_child)

    def update_QN_scale(self, new_scale):
        # Update the Q and the N values self.scaled by the new best reward.
        if self.scale != 0:
            for s in self.QN:
                self.QN[s][0] *= (self.scale / new_scale)

        self.scale = new_scale

    def back_propagate(self, state, action_index, reward):
        """
        Update the Q, N and ucb for all corresponding decedent after a complete rollout
        """

        action = self.grammars[action_index]
        if self.scale != 0:
            self.QN[state + ',' + action][0] += reward / self.scale
        else:
            self.QN[state + ',' + action][0] += 0
        self.QN[state + ',' + action][1] += 1

        while state:
            # print("the state is", state)
            if self.scale != 0:
                self.QN[state][0] += reward / self.scale
            else:
                self.QN[state][0] += 0
            self.QN[state][1] += 1
            self.UCBs[state][self.grammars.index(action)] = self.update_ucb_mcts(state, action)
            if state in self.grammars:
                state = ''
            elif ',' in state:
                state, action = state.rsplit(',', 1)
            else:
                state = ''

    def get_ucb_policy(self, nA):
        """
        Creates an policy based on ucb score.
        """

        def policy_fn(state, node):
            valid_action = self.valid_production_rules(node)

            # collect ucb scores for all valid actions
            policy_valid = []

            sum_ucb = sum([np.exp(self.UCBs[state][a]) for a in valid_action])
            # if all ucb scores identical, return uniform policy
            if len(set(policy_valid)) == 1:
                A = np.zeros(nA)
                A[valid_action] = float(1 / len(valid_action))
                return A

            A = np.zeros(nA, dtype=float)

            for a in valid_action:
                policy_mcts = np.exp(self.UCBs[state][a]) / sum_ucb
                policy_valid.append(policy_mcts)

            best_action = valid_action[np.argmax(policy_valid)]
            A[best_action] += 0.8
            A[valid_action] += float(0.2 / len(valid_action))
            return A

        return policy_fn

    def get_uniform_random_policy(self):
        """
        Creates an random policy to select an unvisited child.
        """

        def policy_fn(UC):
            if len(UC) != len(set(UC)):
                print(UC)
                print(self.grammars)
            action_probs = np.ones(len(UC), dtype=float) * float(1 / len(UC))
            return action_probs

        return policy_fn

    def update_hall_of_fame(self, state, reward, eq):
        """
        If we pass by a concise solution with high score, we store it as an
        single action for future use.
        """
        module = state
        if state.count(',') <= self.max_module:
            if not self.hall_of_fame:
                self.hall_of_fame = [(module, reward, eq)]
            elif eq not in [x[2] for x in self.hall_of_fame]:
                if len(self.hall_of_fame) < self.max_aug:
                    self.hall_of_fame = sorted(self.hall_of_fame + [(module, reward, eq)], key=lambda x: x[1])
                else:
                    if reward > self.hall_of_fame[0][1]:
                        self.hall_of_fame = sorted(self.hall_of_fame[1:] + [(module, reward, eq)], key=lambda x: x[1])

    def MCTS_run(self, num_episodes, num_rollouts=50, verbose=False, print_freq=5, is_first_round=False, reward_threhold=10):
        """
        Monte Carlo Tree Search algorithm
        """

        nA = len(self.grammars)
        states = []

        # The policy we're following:
        # ucb_policy for fully expanded node and uniform_random_policy for not fully expanded node
        ucb_policy = self.get_ucb_policy(nA)
        reward_his = []
        best_solution = ('C', -100)

        for t in range(1, num_episodes + 1):
            print("\tITER {}/{}...".format(t, num_episodes))
            if t % print_freq == 0 and verbose and len(self.hall_of_fame) >= 1:
                print("\tIteration {}/{}...".format(t, num_episodes))
                print("#QN:", len(self.QN.keys()))
                self.print_hofs(-1, verbose=True)
                sys.stdout.flush()
                print([x[1] for x in self.hall_of_fame], reward_threhold)

            if not is_first_round:
                state = 'f->B'
                ntn = ['B']
            else:
                state = 'f->A'
                ntn = ['A']
            unvisited_children = self.get_unvisited_children(state, ntn[0])

            # scenario 1: if current parent node fully expanded, follow ucb_policy
            while not unvisited_children:
                prob = ucb_policy(state, ntn[0])
                print("UCB_policy... prob=", prob)
                action = np.random.choice(np.arange(nA), p=prob / np.sum(prob))
                print('state:', state, '\t action:', self.grammars[action])
                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn[1:])
                if state not in states:
                    states.append(state)

                if not done:
                    state = next_state
                    ntn = ntn_next
                    unvisited_children = self.get_unvisited_children(state, ntn[0])

                    if state.count(',') >= self.max_len:
                        unvisited_children = []
                        self.back_propagate(state, action, 0)
                        reward_his.append(best_solution[1])
                        break
                else:
                    unvisited_children = []
                    if reward > best_solution[1]:
                        self.update_hall_of_fame(next_state, reward, eq)
                        if reward > 0:
                            self.update_QN_scale(reward)
                        best_solution = (eq, reward)
                    # print("BACK-PROPAGATION STEP")
                    self.back_propagate(state, action, reward)
                    reward_his.append(best_solution[1])
                    break

            # scenario 2: if current parent node not fully expanded, follow uniform_random_policy
            while unvisited_children:
                print("uniform_random_policy... ", unvisited_children)
                # prob = uniform_random_policy(unvisited_children)
                action = np.random.choice(unvisited_children)
                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn[1:])
                print('state:', state, '\t action:', self.grammars[action])
                if not done:
                    reward, eq = self.rollout(num_rollouts, next_state, ntn_next)
                    if state not in states:
                        states.append(state)
                if reward > best_solution[1]:
                    self.update_hall_of_fame(next_state, reward, eq)
                    if reward > 0:
                        self.update_QN_scale(reward)
                    best_solution = (eq, reward)
                # 4. BACK-PROPAGATION STEP in MCTS.
                self.back_propagate(state, action, reward)
                reward_his.append(best_solution[1])
                unvisited_children.remove(action)
                if len(self.hall_of_fame) > 1 and max([x[1] for x in self.hall_of_fame]) > reward_threhold:
                    break
            if len(self.hall_of_fame) > 1 and max([x[1] for x in self.hall_of_fame]) > reward_threhold:
                break

        print("#QN:", len(self.QN.keys()))
        self.print_hofs(-1, verbose=True)
        sys.stdout.flush()
        print([x[1] for x in self.hall_of_fame], reward_threhold)

        return reward_his, self.hall_of_fame

    def MCTS_run_orig(self, num_episodes, num_rollouts=50, verbose=False, print_freq=5, is_first_round=False, reward_threhold=10):
        """
        Monte Carlo Tree Search algorithm
        """

        nA = len(self.grammars)
        states = []

        # The policy we're following:
        # ucb_policy for fully expanded node and uniform_random_policy for not fully expanded node
        ucb_policy = self.get_ucb_policy(nA)
        reward_his = []
        best_solution = ('C', -100)

        for t in range(1, num_episodes + 1):
            print("\tITER {}/{}...".format(t, num_episodes))
            if t % print_freq == 0 and verbose and len(self.hall_of_fame) >= 1:
                print("\tIteration {}/{}...".format(t, num_episodes))
                print("#QN:", len(self.QN.keys()))
                self.print_hofs(-2, verbose=True)
                sys.stdout.flush()
                print([x[1] for x in self.hall_of_fame], reward_threhold)

            if not is_first_round:
                state = 'f->B'
                ntn = ['B']
            else:
                state = 'f->A'
                ntn = ['A']
            unvisited_children = self.get_unvisited_children(state, ntn[0])

            # scenario 1: if current parent node fully expanded, follow ucb_policy
            while not unvisited_children:
                prob = ucb_policy(state, ntn[0])
                print("UCB_policy... prob=", prob)
                action = np.random.choice(np.arange(nA), p=prob / np.sum(prob))
                print('state:', state, '\t action:', self.grammars[action])
                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn[1:])
                if state not in states:
                    states.append(state)

                if not done:
                    state = next_state
                    ntn = ntn_next
                    unvisited_children = self.get_unvisited_children(state, ntn[0])

                    if state.count(',') >= self.max_len:
                        unvisited_children = []
                        self.back_propagate(state, action, 0)
                        reward_his.append(best_solution[1])
                        break
                else:
                    unvisited_children = []
                    if reward > best_solution[1]:
                        self.update_hall_of_fame(next_state, reward, eq)
                        if reward > 0:
                            self.update_QN_scale(reward)
                        best_solution = (eq, reward)
                    # print("BACK-PROPAGATION STEP")
                    self.back_propagate(state, action, reward)
                    reward_his.append(best_solution[1])
                    break

            # scenario 2: if current parent node not fully expanded, follow uniform_random_policy
            if len(unvisited_children) != 0:
                print("uniform_random_policy... ", unvisited_children)
                # prob = uniform_random_policy(unvisited_children)
                action = np.random.choice(unvisited_children)
                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn[1:])
                print('state:', state, '\t action:', self.grammars[action])
                if not done:
                    reward, eq = self.rollout(num_rollouts, next_state, ntn_next)
                    if state not in states:
                        states.append(state)
                if reward > best_solution[1]:
                    self.update_hall_of_fame(next_state, reward, eq)
                    if reward > 0:
                        self.update_QN_scale(reward)
                    best_solution = (eq, reward)
                # 4. BACK-PROPAGATION STEP in MCTS.
                self.back_propagate(state, action, reward)
                reward_his.append(best_solution[1])
                unvisited_children.remove(action)
                if len(self.hall_of_fame) > 1 and max([x[1] for x in self.hall_of_fame]) > reward_threhold:
                    break
            if len(self.hall_of_fame) > 1 and max([x[1] for x in self.hall_of_fame]) > reward_threhold:
                break

        return reward_his, self.hall_of_fame

    def print_hofs(self, flag, verbose=False):
        if flag == -1:
            old_vf = copy.copy(self.program.get_vf())
            self.program.vf = [1, ] * self.nvars
            self.task.set_allowed_inputs(self.program.get_vf())
            print("new vf for HOF ranking", self.program.get_vf(), self.task.fixed_column)
        self.task.rand_draw_init_cond()
        print(f"PRINT HOF (free variables={self.task.fixed_column})")
        print("=" * 20)
        for pr in self.hall_of_fame[-len(self.hall_of_fame):]:
            if verbose:
                print('        ' + str(get_state(pr)), end="\n")
                self.print_reward_function_all_metrics(pr[2])
            else:
                print('        ' + str(get_state(pr)), end="\n")
        print("=" * 20)
        if flag == -1:
            self.program.vf = old_vf
            self.task.set_allowed_inputs(old_vf)
            print("reset old vf", self.program.get_vf(), self.task.fixed_column)

    def print_reward_function_all_metrics(self, expr_str):
        """used for print the error for all metrics between the predicted program `p` and true program."""
        y_hat = execute(expr_str, self.task.init_cond.T, self.input_var_Xs)
        dict_of_result = self.task.data_query_oracle._evaluate_all_losses(self.task.init_cond, y_hat)
        # dict_of_result['tree_edit_distance'] = self.task.data_query_oracle.compute_normalized_tree_edit_distance(expr_str)
        print('-' * 30)
        for mertic_name in dict_of_result:
            print(f"{mertic_name} {dict_of_result[mertic_name]}")
        print('-' * 30)


def get_state(pr):
    state_dict = {
        'reward': pr[1],
        'pretty-eq': pretty_print_expr(pr[2]),
        'rules': pr[0],
    }
    return state_dict
