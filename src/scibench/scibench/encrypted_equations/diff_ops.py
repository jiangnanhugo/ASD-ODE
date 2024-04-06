import torch
import torch.nn as nn
import numpy as np


def LaplacianOp(inputs, dx=1.0, dy=1.0):
    '''
            :param inputs: [batch, iH, iW], torch.float
            :return: laplacian of inputs
            '''
    conv_kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float)
    unsqueezed = False
    if inputs.dim() == 2:
        inputs = torch.unsqueeze(inputs, 0)
        unsqueezed = True
    inputs1 = torch.cat([inputs[:, -1:, :], inputs, inputs[:, :1, :]], dim=1)
    inputs2 = torch.cat([inputs1[:, :, -1:], inputs1, inputs1[:, :, :1]], dim=2)
    conv_inputs = torch.unsqueeze(inputs2, dim=1)
    result = torch.nn.functional.conv2d(input=conv_inputs, weight=conv_kernel).squeeze(dim=1) / (dx * dy)
    if unsqueezed:
        result = torch.squeeze(result, 0)
    return result


def lapl(mat, dx, dy):
    """
    Compute the laplacian using `numpy.gradient` twice.
    """
    grad_y, grad_x = np.gradient(mat, dy, dx)
    grad_xx = np.gradient(grad_x, dx, axis=1)
    grad_yy = np.gradient(grad_y, dy, axis=0)
    return grad_xx + grad_yy


class DifferentialOp(nn.Module):
    def __init__(self):
        super(DifferentialOp, self).__init__()
        self.conv_kernel = nn.Parameter(torch.tensor([[[[-1, 0, 1]]]], dtype=torch.float),
                                        requires_grad=False)

    def forward(self, inputs, diffx=False, d=1.0):
        '''
        :param inputs: [batch, iH, iW], torch.float
        :param diffx: if true, compute dc/dx; else, compute dc/dy
        :return:
        '''
        unsqueezed = False
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 0)
            unsqueezed = True
        if diffx:
            inputs = torch.transpose(inputs, -1, -2)
        inputs1 = torch.cat([inputs[:, :, -1:], inputs, inputs[:, :, :1]], dim=2)
        conv_inputs = torch.unsqueeze(inputs1, dim=1)
        result = torch.nn.functional.conv2d(input=conv_inputs, weight=self.conv_kernel).squeeze(dim=1) / (2 * d)
        if diffx:
            result = torch.transpose(result, -1, -2)
        if unsqueezed:
            result = torch.squeeze(result, 0)
        return result
