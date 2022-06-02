import pdb
from copy import deepcopy, copy
from typing import Union, Callable

import torch, numpy as np, psutil


def LIME_fit(X: torch.Tensor, Yp: torch.Tensor):
    """
    Fit affine LIME model as min_{W, b} ||X W + b - Yp||_F^2
    """
    X_ = torch.cat(
        [X, torch.ones(X.shape[:-1] + (1,), dtype=X.dtype, device=X.device)], -1
    )
    W = torch.linalg.lstsq(X_, Yp)[0]
    W, b = W[..., :-1, :], W[..., -1, :]
    return W, b


def sample_around(x0: torch.Tensor, x_std: torch.Tensor, N: int = 100, alf: float = 1e-3):
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


def sample_background(
    dataset: Union[torch.utils.data.Dataset, torch.torch.Tensor, np.ndarray],
    bg_samples,
    dtype=None,
    device=None,
):
    """
    Args:
        dataset: list of x, y pairs or torch dataset
        bg_samples: number of samples from the dataset (randomly)
    Returns:
        samples from the dataset
    """
    if isinstance(dataset, (torch.utils.data.Dataset, list, tuple)):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=psutil.cpu_count(logical=False),
            batch_size=bg_samples,
            shuffle=True,
        )
        X_bg = next(iter(dataloader))[0].to(dtype).to(device)
    else:
        X_bg = torch.as_tensor(
            dataset[torch.randint(0, len(dataset), size=(bg_samples,)), :],
            device=device,
            dtype=dtype,
        )
    return X_bg


def finite_diff(f: Callable, x: torch.Tensor, h: float):
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
