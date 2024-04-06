import numpy as np

import scipy
import sympy
from sympy.parsing import parse_expr

from scibench.sympy2zss_conversion import sympy2zss_module, compute_distance

all_metrics = {
    # Negative mean squared error
    "neg_mse": lambda y, y_hat: -np.mean((y - y_hat) ** 2),
    # Negative root mean squared error
    "neg_rmse": lambda y, y_hat: -np.sqrt(np.mean((y - y_hat) ** 2)),
    # Negative normalized mean squared error
    "neg_nmse": lambda y, y_hat, var_y: -np.mean((y - y_hat) ** 2) / var_y,
    "log_nmse": lambda y, y_hat, var_y: -np.log10(1e-60 + np.mean((y - y_hat) ** 2) / var_y),
    # Negative normalized root mean squared error
    "neg_nrmse": lambda y, y_hat, var_y: -np.sqrt(np.mean((y - y_hat) ** 2) / var_y),
    # (Protected) inverse mean squared error
    "inv_mse": lambda y, y_hat: 1 / (1 + np.mean((y - y_hat) ** 2)),
    # (Protected) inverse normalized mean squared error
    "inv_nmse": lambda y, y_hat, var_y: 1 / (1 + np.mean((y - y_hat) ** 2) / var_y),
    # (Protected) inverse normalized root mean squared error
    "inv_nrmse": lambda y, y_hat, var_y: 1 / (1 + np.sqrt(np.mean((y - y_hat) ** 2) / var_y)),
    # Pearson correlation coefficient       # Range: [0, 1]
    "pearson": lambda y, y_hat: scipy.stats.pearsonr(y, y_hat)[0],
    # Spearman correlation coefficient      # Range: [0, 1]
    "spearman": lambda y, y_hat: scipy.stats.spearmanr(y, y_hat)[0],
    # Accuracy based on R2 value.
    "accuracy(r2)": lambda y, y_hat, var_y, tau: 1 - np.mean((y - y_hat) ** 2) / var_y >= tau,
}


def load_eq_as_tree(one_equation, prints=True):
    try:
        eq_sympy = parse_expr(one_equation)
        eq_sympy = sympy.sympify(str(eq_sympy))
        eq_sympy = eq_sympy.subs(sympy.pi, sympy.pi.evalf()).evalf().factor().simplify().subs(1.0, 1)
        eq_sympy = sympy.sympify(str(eq_sympy))
    except TypeError as te:
        if prints:
            print(te)
            print(f'[{one_equation}]')
        return None, None
    except Exception as e:
        if prints:
            print(e)
            print(f'[{one_equation}]')
        return None, None

    if prints:
        print(f'[{one_equation}]')
        print(f'Eq.: {eq_sympy}')
    return sympy2zss_module(eq_sympy)


def tree_edit_distance(pred_eq_str, gt_eq_str):
    # print(pred_eq_str)
    # print(gt_eq_str)
    gt_eq_tree = load_eq_as_tree(str(gt_eq_str), prints=False)
    est_eq_tree = load_eq_as_tree(pred_eq_str, prints=False)
    ned = compute_distance(est_eq_tree, gt_eq_tree, True)
    ted = compute_distance(est_eq_tree, gt_eq_tree, False)
    # print(f'normalized Edit tree distance (normalized): {ned}\n')
    # print(f'Edit distance: {ted}\n')
    return ned