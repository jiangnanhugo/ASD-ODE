
import numpy as np


from distutils.core import setup
import os

required = [
    "numpy",
    "sympy",
    "scipy",
    "click",
]

setup(name='scibench',
      version='1.0',
      description='Data Oracle for symbolic regression',
      packages=['scibench'],
      setup_requires=["numpy"],
      include_dirs=[np.get_include()],
      install_requires=required
      )


