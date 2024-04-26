from distutils.core import setup

required = [
    "cython",
    "numpy",
    "tensorflow==2.15.0",
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
    description='vectorized context free grammar for symbolic ODE.',
    packages=['grammar'],
    install_requires=required,
)
