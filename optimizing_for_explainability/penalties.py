import pdb
from copy import deepcopy
from pathlib import Path
from typing import Union, Callable

import torch, numpy as np
from importlib.machinery import SourceFileLoader

from .utils import sample_around, sample_background, LIME_fit
from .penalty_functions import mse_penalty, exact_penalty, super_exact_penalty

# import the shap module
shap_path = (
    Path(__file__).absolute().parent.parent / "shap" / "shap" / "__init__.py"
)
shap = SourceFileLoader("shap", str(shap_path)).load_module()


def LIME_penalty(
    penalty_idx: Union[int, torch.Tensor, np.ndarray, list, tuple],
    model: torch.nn.Module,
    dataset: Union[torch.utils.data.Dataset, torch.Tensor, np.ndarray],
    dataset_std: Union[torch.Tensor, np.ndarray, list, float, int],
    penalty_fn: Callable = exact_penalty,
    penalty_scale: float = None,
    test_samples: int = 100,
    bg_samples: int = 1000,
    sample_std: float = 1e-2,
):
    """
    Args:
        model: pytorch model to penalize
        dataset: training dataset
        penalty_fn: penalty function type to apply [MSE, exact, "super"-exact]
        test_samples: number of samples to pick per linear model (one linear model for each background sample)
        bg_samples: number of background samples to create one linear model per
        sample_std: standard deviation of sampling around each sample (already scaled by dataset standard deviation)
    Returns:
        (value of the penalty), (gradients wrt the model parameters)
    """
    param = next(iter(model.parameters()))
    dtype, device = param.dtype, param.device

    X_bg = sample_background(dataset, bg_samples, dtype=dtype, device=device)
    std = torch.as_tensor(dataset_std, device=device, dtype=dtype)
    X_test = sample_around(X_bg, std, N=test_samples, alf=sample_std).transpose(
        0, 1
    )
    W, b = LIME_fit(X_test, model(X_test))
    cstr = penalty_fn(W[..., penalty_idx, 0])

    grads = torch.autograd.grad(cstr, model.parameters())
    for (i, grad) in enumerate(grads):
        if torch.any(torch.isnan(grad)):
            grads[i] = torch.zeros_like(grad)
    grads = [grad.detach() for grad in grads]
    return cstr.detach(), grads


def SHAP_penalty(
    penalty_idx: Union[int, torch.Tensor, np.ndarray, list, tuple],
    model: torch.nn.Module,
    dataset: Union[torch.utils.data.Dataset, torch.Tensor, np.ndarray],
    penalty_fn: Callable = exact_penalty,
    test_samples: int = 10,
    bg_samples: int = 100,
):
    """
    Args:
        model: pytorch model to penalize
        dataset: training dataset
        penalty_fn: penalty function type to apply [MSE, exact, "super"-exact]
        test_samples: number of samples to ask the explainer to give shapley values for (backend does a for loop)
        bg_samples: number of background samples to input into the explainer
    Returns:
        (value of the penalty), (gradients wrt the model parameters)
    """
    param = next(iter(model.parameters()))
    dtype, device = param.dtype, param.device

    X_bg = sample_background(dataset, bg_samples, dtype=dtype, device=device)
    X_test = sample_background(
        dataset, test_samples, dtype=dtype, device=device
    )

    model_cpy = deepcopy(model)
    explainer = shap.DeepExplainer(model_cpy, X_bg)
    values = explainer.shap_values(X_test)

    # explainer_ = shap_original.DeepExplainer(deepcopy(model), X_bg)
    # values_ = explainer_.shap_values(X_test)

    values_norm = values / (torch.linalg.norm(values, dim=-1)[..., None] + 1e-3)
    cstr = penalty_fn(values_norm[..., penalty_idx])

    grads = list(torch.autograd.grad(cstr, model_cpy.parameters()))
    for (i, grad) in enumerate(grads):
        if torch.any(torch.isnan(grad)):
            grads[i] = torch.zeros_like(grad)
    grads = [grad.detach() for grad in grads]
    return cstr.detach(), grads
