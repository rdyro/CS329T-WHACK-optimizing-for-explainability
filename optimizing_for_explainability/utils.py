import pdb
from copy import deepcopy, copy
import torch

from torch import Tensor
from typing import Union, Callable


def lime_fit(X: Tensor, Yp: Tensor):
    """
    Fit affine LIME model as min_{W, b} ||X W + b - Yp||_F^2
    """
    X_ = torch.cat(
        [X, torch.ones(X.shape[:-1] + (1,), dtype=X.dtype, device=X.device)], -1
    )
    W = torch.linalg.lstsq(X_, Yp)[0]
    W, b = W[..., :-1, :], W[..., -1, :]
    return W, b


def sample_around(x0: Tensor, x_std: Tensor, N: int = 100, alf: float = 1e-3):
    """
    Sample points around the x0 points with N samples.
    Args:
        x0:    dataset points to sample around.
        x_std: Per-feature standard deviation of x0 for sampling scaling.
        N:     number of samples (returned as an extra first dimension).
        alf:   scaling of the sampling, scales x_std.

    Returns:
        Output of shape (N,) + x0.shape, N random samples per x0.

    """
    return x0[None, ...] + alf * x_std * torch.randn(
        (N,) + tuple(x0.shape), device=x0.device, dtype=x0.dtype
    )

def finite_diff(f: Callable, x: Tensor, h: float):
    fx, e, Js = f(x), torch.zeros_like(x), []
    for i in range(x.numel()):
        e.zero_()
        e.reshape(-1)[i] = h
        Js.append(torch.clone(((f(x + e) - fx) / h).detach()))
    return torch.stack(Js, -1).reshape(fx.shape + x.shape)

################################################################################


class CatLayer(torch.nn.Module):
    """
    A layer which concatenates the inputs either as a list or separate
    arguments.
    """

    def __init__(self):
        super().__init__()

    def forward(self, *xs):
        if len(xs) == 1:
            return torch.cat(xs[0], -1)
        else:
            return torch.cat(xs, -1)
