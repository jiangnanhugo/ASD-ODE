from scibench.tokens import sciToken
import numpy as np


# these tokens requires to load torch, which will toke a lot of memory.
def LaplacianOp(inputs: np.ndarray, dx=1.0, dy=1.0):
    '''
    :param inputs: [batch, iH, iW], torch.float
    :return: laplacian of inputs
    '''
    import torch
    inputs = torch.from_numpy(inputs).to(torch.double)
    conv_kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.double)
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
    return result.numpy()


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, padding_mode="zeros"):
    """
    Applies a 2D convolution to an array of images. Technically, this function
    computes a correlation instead of a convolution since the kernels are not
    flipped.

    input: numpy array of images with shape = (n, c, h, w)
    weight: numpy array with shape = (c_out, c // groups, kernel_height, kernel_width)
    bias: numpy array of shape (c_out,), default None
    stride: step width of convolution kernel, int or (int, int) tuple, default 1
    padding: padding around images, int or (int, int) tuple or "same", default 0
    dilation: spacing between kernel elements, int or (int, int) tuple, default 1
    groups: split c and c_out into groups to reduce memory usage (must both be divisible), default 1
    padding_mode: "zeros", "reflect", "replicate", "circular", or whatever np.pad supports, default "zeros"

    This function is indended to be similar to PyTorch's conv2d function.
    For more details, see:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    """
    c_out, c_in_by_groups, kh, kw = weight.shape
    kernel_size = (kh, kw)

    if isinstance(stride, int):
        stride = [stride, stride]

    if isinstance(dilation, int):
        dilation = [dilation, dilation]

    if padding:
        input = conv2d_pad(input, padding, padding_mode, stride, dilation, kernel_size)

    n, c_in, h, w = input.shape
    dh, dw = dilation
    sh, sw = stride
    dilated_kh = (kh - 1) * dh + 1
    dilated_kw = (kw - 1) * dw + 1
    assert c_in % groups == 0, f"Number of input channels ({c_in}) not divisible by groups ({groups})."
    assert c_out % groups == 0, f"Number of output channels ({c_out}) not divisible by groups ({groups})."
    c_in_group = c_in // groups
    c_out_group = c_out // groups
    kernel_shape = (c_in_group, dilated_kh, dilated_kw)

    input = input.reshape(n, groups, c_in_group, h, w)
    weight = weight.reshape(groups, c_out_group, c_in_by_groups, kh, kw)

    # Cut out kernel-shaped regions from input
    windows = np.lib.stride_tricks.sliding_window_view(input, kernel_shape, axis=(-3, -2, -1))

    # Apply stride and dilation. Prepare for broadcasting to handle groups.
    windows = windows[:, :, :, ::sh, ::sw, :, ::dh, ::dw]
    weight = weight[np.newaxis, :, :, np.newaxis, np.newaxis, :, :, :]
    h_out, w_out = windows.shape[3:5]

    # Dot product equivalent to either of the next two lines but 10 times faster
    # y = np.einsum("abcdeijk,abcdeijk->abcde", windows, weight)
    # y = (windows * weight).sum(axis=(-3, -2, -1))
    windows = windows.reshape(n, groups, 1, h_out, w_out, c_in_group * kh * kw)
    weight = weight.reshape(1, groups, c_out_group, 1, 1, c_in_group * kh * kw)
    y = np.einsum("abcdei,abcdei->abcde", windows, weight)

    # Concatenate groups as output channels
    y = y.reshape(n, c_out, h_out, w_out)

    if bias is not None:
        y = y + bias.reshape(1, c_out, 1, 1)

    return y


def conv2d_pad(input, padding, padding_mode, stride, dilation, kernel_size):
    if padding == "valid":
        return input

    if padding == "same":
        h, w = input.shape[-2:]
        sh, sw = stride
        dh, dw = dilation
        kh, kw = kernel_size
        ph = (h - 1) * (sh - 1) + (kh - 1) * dh
        pw = (w - 1) * (sw - 1) + (kw - 1) * dw
        ph0 = ph // 2
        ph1 = ph - ph0
        pw0 = pw // 2
        pw1 = pw - pw0
    else:
        if isinstance(padding, int):
            padding = [padding, padding]
        ph0, pw0 = padding
        ph1, pw1 = padding

    pad_width = ((0, 0), (0, 0), (ph0, ph1), (pw0, pw1))

    mode = {
        "zeros": "constant",
        "reflect": "reflect",
        "replicate": "edge",
        "circular": "wrap",
    }.get(padding_mode, padding_mode)

    return np.pad(input, pad_width, mode)


def DifferentialOp(inputs: np.ndarray, diffx=False, d=1.0):
    '''
    :param inputs: [batch, iH, iW], torch.float
    :param diffx: if true, compute dc/dx; else, compute dc/dy
    :return:
    '''
    import torch
    conv_kernel = torch.tensor([[[[-1, 0, 1]]]], dtype=torch.double)
    unsqueezed = False
    if inputs.dim() == 2:
        inputs = torch.unsqueeze(inputs, 0)
        unsqueezed = True
    if diffx:
        inputs = torch.transpose(inputs, -1, -2)
    inputs1 = torch.cat([inputs[:, :, -1:], inputs, inputs[:, :, :1]], dim=2)
    conv_inputs = torch.unsqueeze(inputs1, dim=1)
    result = torch.nn.functional.conv2d(input=conv_inputs, weight=conv_kernel).squeeze(dim=1) / (2 * d)
    if diffx:
        result = torch.transpose(result, -1, -2)
    if unsqueezed:
        result = torch.squeeze(result, 0)
    return result


def ClampOp(inputs: np.ndarray):
    """
    clip the input to [0, 1]
    :param inputs:
    :return:
    """
    import torch
    inputs = torch.from_numpy(inputs).to(torch.double)
    clamped = torch.clamp(inputs, min=0.0, max=1.0)
    return clamped.numpy()


# Annotate unprotected ops
unprotected_ops = [
    # differential operators
    sciToken(LaplacianOp, "laplacian", arity=1, complexity=4),
    sciToken(DifferentialOp, "differential", arity=1, complexity=4),
    sciToken(ClampOp, "clamp", arity=1, complexity=1),
]
