"""Regression task: learn a smooth bowl, y = x1^2 + x2^2 (+ a little noise).

The level sets are bounded rings, so this *needs* hidden width >= 3 to fit well
(width 2 can only fold a rough, faceted approximation) — see docs/.
"""

import torch
import torch.nn as nn

from .base import Task


def make_data(n: int, seed: int):
    """(X, y) with X ~ N(0, I) in 2D and y = x1^2 + x2^2 + small noise."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 2, generator=g)
    y = X[:, 0] ** 2 + X[:, 1] ** 2 + 0.1 * torch.randn(n, generator=g)
    return X, y.unsqueeze(1)


def true_surface(gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    """The noiseless bowl evaluated on a meshgrid."""
    return gx ** 2 + gy ** 2


TASK = Task(
    name="regression",
    make_data=make_data,
    true_surface=true_surface,
    make_criterion=nn.MSELoss,
    # the raw output already *is* the predicted height — nothing to squash.
    activation=lambda pred: pred,
    # bowl heights are >= 0; scale from 0 to the tallest point of the truth.
    surface_range=lambda surf: (0.0, float(surf.max())),
    description="a smooth bowl, y = x1^2 + x2^2 (+ noise)",
)
