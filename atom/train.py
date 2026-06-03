"""Training loop that records a flip-book of the network as it learns.

Every few epochs we freeze a "frame": the current weights, the loss so far,
the gradient sizes, how a probe input flows through the net, and the whole
function the network currently represents over the input plane. The
visualizer turns that flip-book into an animation.
"""

import torch
import torch.nn as nn

from .model import MLP
from .data import make_data, true_surface
from .monitor import NetworkMonitor
from .device import pick_device

# One fixed input we follow through the network in every frame, so the
# activation numbers on the diagram are comparable across time.
PROBE = torch.tensor([[1.0, 1.0]])

# Extent of the 2D plane we render the learned function over.
GRID_LIM = 3.0
GRID_RES = 80


def _input_grid():
    axis = torch.linspace(-GRID_LIM, GRID_LIM, GRID_RES)
    gy, gx = torch.meshgrid(axis, axis, indexing="ij")
    flat = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    return flat, gx, gy


def _frame(model, monitor, epoch, losses, task, grid, gx, gy):
    """Snapshot everything the visualizer needs for one animation frame."""
    device = next(model.parameters()).device
    snap = model.snapshot()
    trace = model.trace(PROBE.to(device))
    with torch.no_grad():
        pred = model(grid).reshape(gx.shape)
        if task == "classification":
            pred = torch.sigmoid(pred)
    return {
        "epoch": epoch,
        "loss": losses[-1],
        "losses": list(losses),
        "sizes": list(model.sizes),
        "snap": snap,
        "trace": trace,
        "arch": model.arch_str(),
        "nparams": model.n_params(),
        "grad_norms": dict(monitor.grad_norms),
        "pred": pred.cpu().clone(),
    }


def train(task="regression", epochs=1000, lr=0.01, n=1000,
          snapshot_every=20, seed=42, device=None):
    """Train the MLP and return (model, history, context).

    history is a list of per-frame snapshots; context holds the static grid
    and the ground-truth surface used for plotting.
    """
    if device is None:
        device = pick_device()
    torch.manual_seed(seed)
    model = MLP(hidden=(2,)).to(device)
    X, y = make_data(n, task, seed)
    X, y = X.to(device), y.to(device)

    criterion = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    monitor = NetworkMonitor().attach(model)
    grid, gx, gy = _input_grid()
    grid = grid.to(device)

    history, losses = [], []
    for epoch in range(epochs):
        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if epoch % snapshot_every == 0 or epoch == epochs - 1:
            history.append(_frame(model, monitor, epoch, losses, task, grid, gx, gy))

    monitor.detach()
    context = {
        "task": task,
        "gx": gx,
        "gy": gy,
        "true_surface": true_surface(gx, gy, task),
        "probe": PROBE,
        "grid_lim": GRID_LIM,
        "total_epochs": epochs,
        "device": str(device),
    }
    return model, history, context
