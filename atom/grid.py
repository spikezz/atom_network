"""A grid of networks, growing in capacity, all learning the same target.

The question this answers: *is the tiny 2-2-1 network too small for the job?*
We lay out a grid of architectures and train them side by side:

    →  across columns: each hidden layer gets one more neuron (wider)
    ↓  down rows:      one more hidden layer is added       (deeper)

The input (2) and output (1) never change — they are fixed by the problem.
Only the hidden layers grow, one neuron / one layer per step.

Every cell is a *full dashboard* — the exact same network diagram + loss curve
+ learned-surface + truth that a single network gets (see atom/visualize.py).
Only the colour scales are shared across the grid instead of repeated per cell.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import torch.nn as nn

from .model import MLP
from .data import make_data, true_surface
from .train import _input_grid, PROBE, GRID_LIM
from .device import pick_device
from .visualize import (scales_from_frames, draw_dashboard, _make_cell_axes,
                        _shared_colorbars, _surface_range)


# --------------------------------------------------------------------------- #
# architectures
# --------------------------------------------------------------------------- #
def make_architectures(rows, cols, base_width=2, base_depth=1):
    """grid[r][c] = tuple of hidden widths.

    Column c -> width = base_width + c (wider).
    Row    r -> depth = base_depth + r (more layers).
    So (0,0) is the original (2,) and each step adds exactly one.
    """
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append(tuple([base_width + c] * (base_depth + r)))
        grid.append(row)
    return grid


# --------------------------------------------------------------------------- #
# training all cells together
# --------------------------------------------------------------------------- #
def _cell_frame(net, epoch, losses, grid, gx, gy, task):
    device = next(net.parameters()).device
    with torch.no_grad():
        pred = net(grid).reshape(gx.shape)
        if task == "classification":
            pred = torch.sigmoid(pred)
    return {
        "epoch": epoch,
        "loss": losses[-1],
        "losses": list(losses),
        "snap": net.snapshot(),
        "trace": net.trace(PROBE.to(device)),
        "pred": pred.cpu().clone(),
        "sizes": list(net.sizes),
        "arch": net.arch_str(),
        "nparams": net.n_params(),
    }


def train_grid(rows=2, cols=2, task="regression", epochs=1500, lr=0.01,
               n=1000, snapshot_every=30, seed=42, device=None):
    """Train a rows×cols grid of MLPs on the same data; return (histories, ctx)."""
    if device is None:
        device = pick_device()
    torch.manual_seed(seed)
    X, y = make_data(n, task, seed)
    X, y = X.to(device), y.to(device)
    arch = make_architectures(rows, cols)

    nets = [[MLP(hidden=arch[r][c]).to(device) for c in range(cols)] for r in range(rows)]
    opt = [[torch.optim.Adam(nets[r][c].parameters(), lr=lr) for c in range(cols)]
           for r in range(rows)]
    criterion = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()

    grid, gx, gy = _input_grid()
    grid = grid.to(device)
    histories = [[[] for _ in range(cols)] for _ in range(rows)]
    losses = [[[] for _ in range(cols)] for _ in range(rows)]

    for epoch in range(epochs):
        for r in range(rows):
            for c in range(cols):
                pred = nets[r][c](X)
                loss = criterion(pred, y)
                opt[r][c].zero_grad()
                loss.backward()
                opt[r][c].step()
                losses[r][c].append(loss.item())
        if epoch % snapshot_every == 0 or epoch == epochs - 1:
            for r in range(rows):
                for c in range(cols):
                    histories[r][c].append(
                        _cell_frame(nets[r][c], epoch, losses[r][c],
                                    grid, gx, gy, task))

    context = {
        "task": task, "rows": rows, "cols": cols, "arch": arch,
        "gx": gx, "gy": gy, "true_surface": true_surface(gx, gy, task),
        "grid_lim": GRID_LIM, "total_epochs": epochs, "device": str(device),
    }
    return histories, context


# --------------------------------------------------------------------------- #
# rendering the grid (every cell is a full dashboard)
# --------------------------------------------------------------------------- #
def render_grid(histories, context, out_gif="outputs/grid.gif",
                out_png="outputs/grid.png", fps=12):
    rows, cols = context["rows"], context["cols"]
    task = context["task"]
    total = context["total_epochs"]
    n_frames = len(histories[0][0])

    all_frames = [f for row in histories for cell in row for f in cell]
    w_norm, a_norm = scales_from_frames(all_frames)
    s_vmin, s_vmax = _surface_range(context)

    fig = plt.figure(figsize=(9.6 * cols + 1.4, 5.3 * rows + 0.9))
    outer = fig.add_gridspec(rows, cols, left=0.035, right=0.90,
                             top=0.90, bottom=0.07, hspace=0.42, wspace=0.14)
    cell_axes = [[_make_cell_axes(fig, outer[r, c]) for c in range(cols)]
                 for r in range(rows)]
    _shared_colorbars(fig, w_norm, s_vmin, s_vmax)
    fig.text(0.5, 0.015,
             "each cell is a full dashboard  ·  lines = weights (red + / blue −, "
             "thicker = larger, value on line)  ·  circles = neurons (fill = "
             "activation, number inside, b = bias)",
             ha="center", va="bottom", fontsize=8.5, color="#555")

    def update(i):
        for r in range(rows):
            for c in range(cols):
                f = histories[r][c][i]
                draw_dashboard(cell_axes[r][c], f, context, w_norm, a_norm,
                               s_vmin, s_vmax,
                               net_title=f"{f['arch']}   ·   {f['nparams']} params")
        epoch = histories[0][0][i]["epoch"]
        fig.suptitle(
            f"atom_network — does more capacity help?   ·   {task}   ·   "
            f"→ wider hidden layers,  ↓ deeper   ·   epoch {epoch}/{total}",
            fontsize=15, weight="bold")
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    update(n_frames - 1)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_gif, out_png
