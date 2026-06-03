"""Synthetic datasets — small, 2D, and easy to picture.

Both tasks live on a 2D input plane, which is exactly why we can draw the
function the network has learned as a flat heatmap and compare it to the
truth side by side.
"""

import torch


def make_data(n: int = 1000, task: str = "regression", seed: int = 42):
    """Return (X, y) with X of shape (n, 2) and y of shape (n, 1).

    - "regression": a smooth bowl, y = x1^2 + x2^2 (+ a little noise).
    - "classification": the classic XOR-like split, label 1 when x1 and x2
      share a sign. Not linearly separable — it *needs* the hidden layer.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 2, generator=g)

    if task == "regression":
        y = X[:, 0] ** 2 + X[:, 1] ** 2 + 0.1 * torch.randn(n, generator=g)
        y = y.unsqueeze(1)
    elif task == "classification":
        y = ((X[:, 0] * X[:, 1]) > 0).float().unsqueeze(1)
    else:
        raise ValueError(f"unknown task: {task!r}")

    return X, y


def true_surface(gx: torch.Tensor, gy: torch.Tensor, task: str) -> torch.Tensor:
    """The ground-truth target evaluated on a meshgrid — what the network
    is *trying* to become."""
    if task == "regression":
        return gx ** 2 + gy ** 2
    if task == "classification":
        return ((gx * gy) > 0).float()
    raise ValueError(f"unknown task: {task!r}")
