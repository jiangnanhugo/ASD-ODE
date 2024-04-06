## 0. Prequisites

### 0.1 Dependency packages

```bash
pip install Cython
pip install cryptography
```


### 1. Instructions to use the data oracle
The following will be added to the paper appendix in the next revision.

When we evaluate a new program in the `main.py`, we will execute the following command line argument:
```bash
python main.py --input_eq_name PATH_TO_THE_INPUT_SYMBOLIC_EQUATION \
--time_limit MAX_QUERY_TIME
--output_filename ../PATH_TO_THE_OUTPUT_FILE \ # save your predicted expression into this file
```
Here, 
    - `input_eq_name` contains the file name storing the ground-truth equation and other information such as the type and amount of noise used during evaluation, etc. The ground-truth equation is used by the oracle to generate the output [will be discussed later]. The input file is the ground-truth equation. 

    - `time_limit` is used to let your program know the time limit of the evaluation. The unit is in seconds. Your program can design different strategies for comparison with different time limits. However, the oracle will be killed once the time limit has been reached, whether the program decides to quit itself or not.

    - `output_filename` designates the file you will need to write the symbolic equation you have found into.

#### Data Oracle `Equation_evaluator`
- Your Python package will need to access the Oracle by calling the evaluate method from an object of the class `Equation_evaluator`. In your `main.py`, you will need to initialize an `Equation_evaluator` object. You can do this in the following way. First in the header of `main.py`:
```python
from symbolic_equation_evaluator import *
```
Then in the main program, execute:
```python
Eq_eval = Equation_evaluator(input_eq_name)
```
Where the `input_eq_name` denotes the input file name passed from the command line argument.


- `Equation_evaluator` provides you with this method:
```python
def evaluate(self, X):
```
This method can be used as the oracle. When queried by the input $X$, it returns noisy estimations of $f(X)$. Here, The datatype of $X$ is `numpy.ndarray`. It is a matrix. The first dimension corresponds to the batch size and the second dimension corresponds to `number_of_variables`. Basically, each row of $X$ represents one input to the symbolic expression. The output of evaluate will be a vector, where the $i$-th output is the noisy estimation of $f(X)_i$.

- Other useful functions of the Equation_evaluator class are:
```python
def get_nvars(self):
```
This function returns the number of input variables of the symbolic equation.
```python
def get_function_set(self):
```
This function returns the set of operators possibly used in the symbolic equation. An example output looks like:
```python
{'sin': 1, 'cos': 1, 'add': 2, 'sub' : 2, 'mul' : 2, 'div': 2, 'const': 0}
```
This output means the ground-truth equation may contain `add, sub, mul, div` as binary operators, `sin` and `cos` functions as unary operators, and real-valued constants. The numbers as the values of the dictionary denote the arity of the operators. The full set of operators we consider are `[sin, cos, exp, log, pow, inv, add, sub, mul, div]`. 

#### Output 
- You will need to write the symbolic equation that you found into the output file. The output file contains only one line, the pre-order of the equation you have found. For example, the following line can be one valid equation in the output file:
```python
['add', 'mul',  '0.1',  'X_1', 'sin',  'X_2']
```
This represents the equation `0.1*X1 + sin(X2)`.