import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from base import KnownEquation, LogUniformSampling, IntegerUniformSampling, UniformSampling, LogUniformSampling2d

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
import os
import sympy
from sympy import MatrixSymbol, Matrix, Symbol
from diff_ops import LaplacianOp, DifferentialOp

EQUATION_CLASS_DICT = OrderedDict()


def register_eq_class(cls):
    EQUATION_CLASS_DICT[cls.__name__] = cls
    return cls


def get_eq_obj(key, **kwargs):
    if key in EQUATION_CLASS_DICT:
        return EQUATION_CLASS_DICT[key](**kwargs)
    raise KeyError(f'`{key}` is not expected as a equation object key')


@register_eq_class
class GrainGrowth64x64(KnownEquation):
    _eq_name = 'Grain_Growth_64x64'
    _function_set = ['add', 'sub', 'mul', 'n2', 'n3', 'laplacian', 'const']
    expr_obj_thres = 0.01
    expr_consts_thres = None

    # simulated_exec = True

    def __init__(self, n_grains=10):
        self.A = 1  # np.random.randn(1)[0]
        self.B = 1  # np.random.randn(1)[0]
        # self.kappa = np.random.randn(1)[0]  # .to(device)

        self.lap = LaplacianOp()

        self.dx = 0.5
        self.dy = 0.5
        self.Nx = 64
        self.Ny = 64
        self.dim = [(self.Nx, self.Ny), ]

        self.dt = 1e-2

        vars_range_and_types = [LogUniformSampling2d(1e-3, 1.0, only_positive=True, dim=(self.Nx, self.Ny)) for i in range(n_grains)]
        # self.x = [MatrixSymbol('X_0', self.Nx, self.Ny)]
        super().__init__(num_vars=n_grains, vars_range_and_types=vars_range_and_types)
        etas = self.x

        self.L = 5.0  # nn.Parameter(torch.randn(1) * 5 + 0.1, requires_grad=True)
        self.kappa = 0.1  # nn.Parameter(torch.randn(1) * 5 + 0.1, requires_grad=True)

        self.lap = LaplacianOp()
        sum_eta_2 = sum(eta ** 2 for eta in etas)

        etas_new = [None] * n_grains
        laplacian = sympy.Function('Laplacian')
        for i in range(n_grains):
            dfdeta = -self.A * etas[i] + self.B * (etas[i]) ** 3

            sq_sum = sum_eta_2 - (etas[i]) ** 2
            dfdeta += 2 * etas[i] * sq_sum
            term1 = dfdeta - self.kappa * laplacian(etas[i])  # self.lap(etas[i], self.dx, self.dy)
            etas_new[i] = self.L * term1
            print(etas_new[i])
        self.sympy_eq = etas_new


# @register_eq_class
class SpinodalDecomp64x64(KnownEquation):
    _eq_name = 'Spinodal_Decomposition_64x64'
    _function_set = ['add', 'sub', 'mul', 'div', 'clamp', 'laplacian', 'const']
    expr_obj_thres = 0.01
    expr_consts_thres = None
    simulated_exec = True

    def __init__(self):
        # super(SpinodalDecomp, self).__init__()
        # c is the input matrix; A, M, kappa is the constants in the expressions
        self.A = 1
        self.M = 1
        self.kappa = 0.5

        self.lap = LaplacianOp()
        self.diff = DifferentialOp()

        self.dx = 1
        self.dy = 1
        self.Nx = 64
        self.Ny = 64
        self.dim = [(self.Nx, self.Ny), ]

        self.dt = 1e-2

        vars_range_and_types = [LogUniformSampling2d(1e-3, 1.0, only_positive=True, dim=(self.Nx, self.Ny))]
        super().__init__(num_vars=1, vars_range_and_types=vars_range_and_types)
        self.x = [MatrixSymbol('X0', self.Nx, self.Ny)]
        c = self.x

        self.torch_func = self.forward
        self.sympy_eq = "EMPTY"
        ### consts = [0:1.0, 1:2.0, 2:A(1), 3:kappa(0.5), 4:dt(1e-2), 5:M(1)]
        consts = [1.0, 2.0, self.A, self.kappa, self.dt, self.M]

        ### tree1 = 2 * self.A * c * (1 - c) * (1 - 2*c)
        tree1 = [(2, "mul"), (0, 1), (2, "mul"), (0, 2), (2, "mul"), (1, 0), (2, "mul"), (2, "sub"), (0, 0), (1, 0), (2, "sub"),
                 (0, 0), (2, "mul"), (0, 1), (1, 0)]
        ### tree2 = self.kappa * self.lap(c, self.dx, self.dy)
        tree2 = [(2, "mul"), (0, 3), (2, "laplacian"), (1, 0)]
        # deltaF = 2 * self.A * c * (1-c) * (1-2*c) - self.kappa * self.lap(c, self.dx, self.dy)
        deltaF = [(2, "sub")]
        deltaF.extend(tree1)
        deltaF.extend(tree2)
        # dc = self.dt * self.lap(self.M*deltaF, self.dx, self.dy)
        dc = [(2, "mul"), (0, 4), (2, "laplacian"), (2, "mul"), (0, 5)]
        dc.extend(deltaF)
        # c_new = torch.clamp(c + dc, min=0.0001, max=0.9999)
        preorder_traversal = [(2, "clamp"), (2, "add"), (1, 0)]
        preorder_traversal.extend(dc)
        self.preorder_traversal = []
        for x in preorder_traversal:
            if x[0] == 0:
                self.preorder_traversal.append((consts[x[1]], "const"))
            elif x[0] == 1:
                self.preorder_traversal.append((str(c[x[1]]), "var"))
            elif x[0] == 2:
                if x[1] in ['mul', 'sub', 'div', 'add']:
                    self.preorder_traversal.append((x[1], "binary"))
                elif x[1] == "laplacian":
                    self.preorder_traversal.append(("laplacian", "unary"))
                elif x[1] == "clamp":
                    self.preorder_traversal.append((x[1], "unary"))

    def forward(self, c):
        # equation (4:18) + (4:17)
        deltaF = 2 * self.A * c * (1 - c) * (1 - 2 * c) - self.kappa * self.lap(c, self.dx, self.dy)
        # equation (4.16)
        dc = self.dt * self.lap(self.M * deltaF, self.dx, self.dy)
        # c_new = c+dc
        c_new = torch.clamp(c + dc, min=0, max=1)
        return c_new
