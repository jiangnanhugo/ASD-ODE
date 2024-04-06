import sympy
from sympy.utilities.misc import func_name
from zss import Node, simple_distance

NODE_WITH_END_TOKEN_SET = set()
END_OF_TREE_TOKEN = 'EOT'


def preorder_traverse_tree(node, symbol_list, returns_binary_tree=False):
    if node.is_number:
        symbol = 'c'
    elif isinstance(node, sympy.Symbol):
        symbol = str(node)
    else:
        symbol = func_name(node)
        if returns_binary_tree and symbol in NODE_WITH_END_TOKEN_SET:
            symbol = f'Bi{symbol}'

    symbol_list.append(symbol)
    num_children = len(node.args)
    for i, child_node in enumerate(node.args):
        if returns_binary_tree and symbol in NODE_WITH_END_TOKEN_SET and 0 < i < num_children - 1:
            symbol_list.append(symbol)
        preorder_traverse_tree(child_node, symbol_list, returns_binary_tree=returns_binary_tree)
    if not returns_binary_tree and symbol in NODE_WITH_END_TOKEN_SET:
        symbol_list.append(END_OF_TREE_TOKEN)


def sympy2zss_module(current_sympy_eq, parent_node=None, node_list=None):
    if node_list is None:
        node_list = list()

    if current_sympy_eq.is_number:
        node_label = 'Const'
    elif isinstance(current_sympy_eq, sympy.Symbol):
        node_label = str(current_sympy_eq)
    else:
        node_label = func_name(current_sympy_eq)

    current_idx = len(node_list)
    current_node = Node(node_label)
    if parent_node is not None:
        parent_node.addkid(current_node)

    node_list.append(current_idx)
    for child_node in current_sympy_eq.args:
        sympy2zss_module(child_node, current_node, node_list)
    return current_node


def count_nodes(zss_node):
    if zss_node is None:
        return 0

    count = 1
    for child in zss_node.children:
        count += count_nodes(child)
    return count


def compute_distance(est_eq_tree, gt_eq_tree, normalizes=False):
    edit_dist = simple_distance(est_eq_tree, gt_eq_tree)
    if not normalizes:
        return edit_dist

    num_gt_nodes = count_nodes(gt_eq_tree)
    return min([edit_dist, num_gt_nodes]) / num_gt_nodes
