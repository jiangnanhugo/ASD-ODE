from .base import EQUATION_CLASS_DICT, get_eq_obj
from .equation_odes_sindy import *


def equation_object_loader(equation_name):
    for eqname in EQUATION_CLASS_DICT:

        one_equation = get_eq_obj(eqname)

        if equation_name == one_equation._eq_name:
            return one_equation()
    print(f"input equation {equation_name} not found")
    return None
