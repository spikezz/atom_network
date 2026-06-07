"""Classification task: the classic XOR-like split.

Label is 1 when x1 and x2 share a sign (quadrants 1 & 3), 0 otherwise. The
decision boundary is the two axes crossing — not linearly separable, so it
*needs* the hidden layer. The net outputs a logit; we train with BCE and display
the sigmoid probability.
"""

import torch
import torch.nn as nn

from .base import Task


def make_data(n: int, seed: int):
    """(X, y) with X ~ N(0, I) in 2D and y = 1 where x1 and x2 share a sign."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 2, generator=g)
    y = ((X[:, 0] * X[:, 1]) > 0).float()
    return X, y.unsqueeze(1)


def true_surface(gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    """The XOR checkerboard evaluated on a meshgrid (1 in quadrants 1 & 3)."""
    return ((gx * gy) > 0).float()


TASK = Task(
    name="classification",
    make_data=make_data,
    true_surface=true_surface,
    make_criterion=nn.BCEWithLogitsLoss,
    # the net emits a logit; squash to a probability for the learned surface.
    activation=torch.sigmoid,
    # probabilities live in [0, 1].
    surface_range=lambda surf: (0.0, 1.0),
    description="XOR-like split, label 1 when x1 and x2 share a sign",
)
