from setuptools import setup
import numpy as np
from Cython.Build import cythonize


from distutils.core import setup
import os

required = [
    "cython",
    "numpy",
    "sympy",
    "scipy",
    "click",
    "cryptography",
    "zss"
]

setup(name='scibench',
      version='1.0',
      description='Data Oracle for symbolic regression',
      packages=['scibench'],
      setup_requires=["numpy", "Cython"],
      ext_modules=cythonize([os.path.join('scibench', 'cyfunc.pyx')]),
      include_dirs=[np.get_include()],
      install_requires=required
      )


