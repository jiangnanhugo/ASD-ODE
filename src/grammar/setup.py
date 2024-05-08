from distutils.core import setup

required = [
    "cython",
    "numpy",
    "numba",
    "sympy",
    "click",
    "tqdm",
    "commentjson",
    "PyYAML",
    "pathos"
]

setup(
    name='grammar',
    version='1.1',
    description='context free grammar for symbolic ODE.',
    packages=['grammar'],
    install_requires=required,
)
