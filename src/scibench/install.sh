export CFLAGS="-I $(python -c "import numpy; print(numpy.get_include())") $CFLAGS"
CC=gcc python setup.py build_ext --inplace
