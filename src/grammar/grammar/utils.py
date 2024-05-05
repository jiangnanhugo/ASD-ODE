import sympy
from sympy.core.numbers import Float, Rational, NegativeOne, Integer
from sympy import simplify, expand, Symbol
from sympy.parsing.sympy_parser import parse_expr


def pretty_print_expr(eq) -> str:
    '''
    ask sympy simplify to pretty print the expression.
    '''
    if type(eq) == str:
        eq = parse_expr(eq)
    return str(expand(simplify(eq)))


def nth_repl(s, sub, repl, n):
    find = s.find(sub)
    # If find is not -1 we have found at least one match for the substring
    i = find != -1
    # loop util we find the nth or we find no match
    while find != -1 and i != n:
        # find + 1 means we start searching from after the last match
        find = s.find(sub, find + 1)
        i += 1
    # If i is equal to n we found nth match so replace
    if i == n:
        return s[:find] + repl + s[find + len(sub):]
    return s


def expression_to_template(expr, stand_alone_constants) -> str:
    C = Symbol('C')
    all_floats = list(expr.atoms(Float))
    for fi in all_floats:
        if len(stand_alone_constants) >= 1 and min([abs(fi - ci) for ci in stand_alone_constants]) < 1e-5:
            continue
        expr = expr.replace(fi, C)
    return str(expr)
