"""What a *task* is, in one place.

A task is everything that changes when you swap "learn a smooth bowl" for "learn
the XOR split" — the data, the ground truth, the loss, and a couple of display
choices. Bundling them into one object means the rest of the codebase (training,
the grid, the visualizer) never branches on a task name again: it just asks the
task to do these things.

Add a new task by dropping a `mytask.py` into this package that defines a
module-level `TASK = Task(...)`. Nothing else needs to change — `atom/tasks/
__init__.py` discovers it by filename, and the CLIs pick it up automatically.
"""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Task:
    """A self-contained learning problem on the 2D input plane.

    Fields are small callables rather than methods so a task file reads as a flat
    list of "here is the data, here is the loss, ..." — easy for a learner to
    scan. `make_criterion` returns a fresh loss module per call (criterions are
    stateless here, but this keeps training loops free to own their own).
    """

    name: str
    # (n, seed) -> (X[n,2], y[n,1]); owns its RNG so runs are reproducible.
    make_data: Callable[[int, int], tuple]
    # (gx, gy) meshgrid -> the target surface the network is trying to become.
    true_surface: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    # () -> the loss module to train with (MSE for regression, BCE for XOR, ...).
    make_criterion: Callable[[], nn.Module] = nn.MSELoss
    # raw net output -> the quantity we *display* as the learned surface. For
    # regression that's the output itself; for a logit classifier it's sigmoid.
    activation: Callable[[torch.Tensor], torch.Tensor] = lambda pred: pred
    # the target surface -> (vmin, vmax) colour range shared across the grid.
    surface_range: Callable[[torch.Tensor], tuple] = (
        lambda surf: (0.0, float(surf.max())))
    # one-line human description (shown in --help / docs).
    description: str = ""
