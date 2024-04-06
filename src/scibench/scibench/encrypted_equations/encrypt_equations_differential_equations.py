#### Requirement:
import json
import os

import numpy as np
import xxhash
from typing import List

from cryptography.fernet import Fernet

import sympy
from sympy import *


def generate_new_key(saveto_filename):
    key = Fernet.generate_key()

    # string the key in a file
    with open(saveto_filename, 'wb') as filekey:
        filekey.write(key)


def encrypt_equation(equation, output_eq_file, key_filename=None, is_encrypted=0):
    """
    {eq_name: "", n_vars: , eq_expression: {}}
    """
    if is_encrypted == 1:
        # opening the key
        with open(key_filename, 'rb') as filekey:
            key = filekey.read()

        # using the generated key
        fernet = Fernet(key)
        # encrypting the Sympy Equation
        encrypted = fernet.encrypt(equation)
        with open(output_eq_file, 'wb') as encrypted_file:
            encrypted_file.write(b'1\n')
            encrypted_file.write(encrypted)
    else:
        with open(output_eq_file, 'wb') as encrypted_file:
            encrypted_file.write(b'0\n')
            encrypted_file.write(equation)


def to_binary_expr_tree(expr):
    """convert a Sympy expression to a binary expression tree"""
    if isinstance(expr, Symbol):
        return [str(expr)]
    elif isinstance(expr, Float) or isinstance(expr, Integer) or isinstance(expr, Rational):
        return [expr]
    elif expr == sympy.pi:
        return np.pi
    elif expr == sympy.EulerGamma:
        return np.euler_gamma
    else:
        op = expr.func
        args = expr.args

        if len(args) <= 2:
            return [op.__name__] + [to_binary_expr_tree(arg) for arg in args]
        else:
            left = to_binary_expr_tree(args[0])
            right = to_binary_expr_tree(op(*args[1:]))
            return [op.__name__, left, right]


def is_float(s):
    """Determine whether the input variable can be cast to float."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def symbolic_equation_to_preorder_traversal(expr) -> List:
    def flatten(S):
        if S == []:
            return S
        if isinstance(S[0], list):
            return flatten(S[0]) + flatten(S[1:])
        return S[:1] + flatten(S[1:])
    binary_tree=to_binary_expr_tree(expr)

    preorder_traversal_expr = flatten(binary_tree)
    preorder_traversal_tuple = []
    for idx, it in enumerate(preorder_traversal_expr):
        if is_float(it):
            preorder_traversal_tuple.append((it, 'const'))
        elif it.startswith('x') or it.startswith('X'):
            preorder_traversal_tuple.append((it, 'var'))
        elif it in ['add', 'Add', 'mul', 'Mul', 'sub', 'Sub', 'div', 'Div', 'pow', 'Pow']:
            preorder_traversal_tuple.append((it.lower(), 'binary'))
        elif it in ['inv', 'Inv', 'sqrt', 'Sqrt', 'sin', 'Sin', 'cos', 'Cos', 'exp', 'Exp', 'log', 'Log', 'n2', 'n3', 'n4']:
            preorder_traversal_tuple.append((it.lower(), 'unary'))
    return preorder_traversal_tuple


def main(private_key_folder='./', key_filename="public.key", output_folder="./", folder_prefix='equation_family'):
    if not os.path.isfile(os.path.join(private_key_folder, key_filename)):
        print('A new key is generated!')
        generate_new_key(key_filename)
    name_map = {}
    for eqname in EQUATION_CLASS_DICT:

        one_equation = get_eq_obj(eqname)
        print(len(one_equation.sympy_eq))
        for idx, one_sympy_eq in enumerate(one_equation.sympy_eq):
            print(idx, one_sympy_eq)
            preorder_traversal = symbolic_equation_to_preorder_traversal(one_sympy_eq)

            if hasattr(one_equation, 'expr_obj_thres'):
                expr_obj_thres = one_equation.expr_obj_thres
            else:
                expr_obj_thres = 0.01
            name_map[one_equation._eq_name] = eqname

            equation = {"eq_name": one_equation._eq_name+f"_d{idx}",
                        "num_vars": one_equation.num_vars,
                        "dim": one_equation.dim,
                        "vars_range_and_types": one_equation.vars_range_and_types_to_json_str(),
                        "function_set": one_equation._function_set,
                        "eq_expression": str(preorder_traversal),
                        "simulated_exec": one_equation.simulated_exec,
                        "expr": str(one_sympy_eq),
                        "expr_obj_thres": expr_obj_thres}

            user_encode_data = json.dumps(equation).encode('utf-8')
            if not os.path.isdir(os.path.join(output_folder, folder_prefix)):
                os.makedirs(os.path.join(output_folder, folder_prefix))
            output_eq_file = os.path.join(output_folder, folder_prefix, eqname + f"_d{idx}" + ".in")

            encrypt_equation(user_encode_data, output_eq_file, is_encrypted=0)
    for name in name_map:
        print(name, name_map[name])


if __name__ == '__main__':
    X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9, X_10, X_11, X_12 = symbols('X0,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12')
    # from equation_odes import *

    # main(output_folder='/home/jiangnan/PycharmProjects/cvdso/data/', folder_prefix='sindy_data')
    from equation_pde_materials import *

    main(output_folder='/home/jiangnan/PycharmProjects/cvdso/data/', folder_prefix='sindy_data')