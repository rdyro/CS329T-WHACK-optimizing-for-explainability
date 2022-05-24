import pdb
from copy import deepcopy, copy
import torch


def lime_fit(X, Yp):
    X_ = torch.cat(
        [X, torch.ones(X.shape[:-1] + (1,), dtype=X.dtype, device=X.device)], -1
    )
    W = torch.linalg.lstsq(X_, Yp)[0]
    W, b = W[..., :-1, :], W[..., -1, :]
    return W, b


def sample_around(x0, x_std, N=100, alf=1e-3):
    opts = dict(device=x0.device, dtype=x0.dtype)
    return x0[None, ...] + alf * x_std * torch.randn(
        (N,) + tuple(x0.shape), **opts
    )


################################################################################
