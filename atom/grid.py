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

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import torch.nn as nn

from .model import MLP
from .data import make_data, true_surface
from .train import _input_grid, PROBE, GRID_LIM
from .device import pick_device
from .visualize import (scales_from_frames, draw_dashboard, _make_cell_axes,
                        _shared_colorbars, _surface_range, render)


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
def _cell_frame(net, epoch, losses, grid, gx, gy, task, lr, seed):
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
        "lr": lr,
        "seed": seed,
    }


def train_grid(rows=2, cols=2, task="regression", epochs=1500, lr=0.01,
               n=1000, snapshot_every=30, seed=42, device=None,
               base_width=2, base_depth=1):
    """Train a rows×cols grid of MLPs on the same data; return (histories, ctx)."""
    if device is None:
        device = pick_device()
    torch.manual_seed(seed)
    X, y = make_data(n, task, seed)
    X, y = X.to(device), y.to(device)
    arch = make_architectures(rows, cols, base_width, base_depth)

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
                                    grid, gx, gy, task, lr, seed))

    context = {
        "task": task, "rows": rows, "cols": cols, "arch": arch,
        "gx": gx, "gy": gy, "true_surface": true_surface(gx, gy, task),
        "grid_lim": GRID_LIM, "total_epochs": epochs, "device": str(device),
        # everything a later --rerun needs to retrain a single cell identically:
        # same data + same architecture grid, only a fresh init seed.
        "base_width": base_width, "base_depth": base_depth,
        "lr": lr, "n": n, "snapshot_every": snapshot_every, "seed": seed,
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


# --------------------------------------------------------------------------- #
# rendering a whole run: the grid + every cell as its own standalone dashboard
# --------------------------------------------------------------------------- #
def _cell_dir(run_dir, r, c, arch):
    """The subfolder a given cell's standalone dashboard lives in."""
    return Path(run_dir) / f"r{r}c{c}_{arch}"


def _render_cell(history, context, run_dir, r, c, fps=12):
    """Render one cell as its own standalone dashboard (gif + png) into its
    subfolder, exactly like a single network. Returns the subfolder path."""
    arch = history[-1]["arch"]
    cell_dir = _cell_dir(run_dir, r, c, arch)
    cell_dir.mkdir(parents=True, exist_ok=True)
    render(history, context,
           out_gif=str(cell_dir / f"{arch}.gif"),
           out_png=str(cell_dir / f"{arch}.png"), fps=fps)
    return cell_dir


def _save_state(run_dir, histories, context):
    """Persist the full run (all cell flip-books + context) so a later
    ``--rerun`` can replace individual cells without retraining the good ones."""
    with open(Path(run_dir) / "run_state.pkl", "wb") as f:
        pickle.dump({"histories": histories, "context": context}, f)


def render_run(histories, context, run_dir, fps=12):
    """Render one complete run into a single folder.

    Produces, under ``run_dir``:
      * ``grid.gif`` / ``grid.png``           — the overall grid (all cells)
      * ``<r{r}c{c}_{arch}>/<arch>.gif|.png`` — every cell, on its own, rendered
        as the *exact same* standalone dashboard a single network gets.
      * ``run_state.pkl``                     — the recorded run, so individual
        cells can be re-trained later with ``--rerun`` (see ``rerun_run``).

    Each cell's history is already a normal single-net flip-book, so we just
    hand it to the single-net ``render()`` — no special-casing, no detail lost.
    Returns (run_dir, [cell_dir, ...]).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rows, cols = context["rows"], context["cols"]

    # 1) the overall grid (total figure + animation)
    render_grid(histories, context,
                out_gif=str(run_dir / "grid.gif"),
                out_png=str(run_dir / "grid.png"), fps=fps)

    # 2) every subplot on its own, in its own subfolder
    cell_dirs = []
    for r in range(rows):
        for c in range(cols):
            cell_dirs.append(_render_cell(histories[r][c], context,
                                          run_dir, r, c, fps))

    # 3) remember the run so failed cells can be retried later
    _save_state(run_dir, histories, context)
    return run_dir, cell_dirs


# --------------------------------------------------------------------------- #
# re-running only the cells that failed — keep the good cells untouched
# --------------------------------------------------------------------------- #
def rerun_cells(context, cells, init_seed, device=None, lr=None):
    """Re-train just the listed (row, col) cells on the *same data* the grid
    used, but from a *fresh init seed* so a cell stuck in a bad minimum gets a
    real second chance. Returns {(r, c): history}.

    ``lr`` overrides the learning rate for this retry; None reuses the rate the
    original grid trained with. Everything that would change the frame count
    (epochs, snapshot_every) stays locked to the run so the cell still lines up
    with its neighbours in the grid animation.

    Nothing here touches the other cells — only the ones asked for are trained.
    """
    if device is None:
        device = pick_device()
    task = context["task"]
    epochs = context["total_epochs"]
    lr = context["lr"] if lr is None else lr
    snapshot_every = context["snapshot_every"]
    arch_grid = context["arch"]
    gx, gy = context["gx"], context["gy"]

    # identical dataset to the original grid run (same seed) ...
    X, y = make_data(context["n"], task, context["seed"])
    X, y = X.to(device), y.to(device)
    grid, _, _ = _input_grid()
    grid = grid.to(device)
    criterion = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()

    out = {}
    for (r, c) in cells:
        hidden = tuple(arch_grid[r][c])
        torch.manual_seed(init_seed)          # ... but a fresh init each rerun
        net = MLP(hidden=hidden).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        losses, history = [], []
        for epoch in range(epochs):
            pred = net(X)
            loss = criterion(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if epoch % snapshot_every == 0 or epoch == epochs - 1:
                history.append(_cell_frame(net, epoch, losses, grid, gx, gy, task,
                                           lr, init_seed))
        out[(r, c)] = history
    return out


def rerun_run(run_dir, cells, init_seed, fps=12, device=None, lr=None):
    """Retrain the listed cells inside an existing run folder and replace them
    *in place*: each retried cell's own dashboard is re-rendered, and the grid
    overview is refreshed so it shows the new result — every other cell's
    standalone dashboard is left exactly as it was.

    ``lr`` overrides the learning rate for the retry (None = the run's own rate).

    Returns {(r, c): last_frame} for reporting the new loss.
    """
    run_dir = Path(run_dir)
    state_path = run_dir / "run_state.pkl"
    if not state_path.exists():
        raise FileNotFoundError(
            f"{state_path} not found — this run predates --rerun support; "
            f"re-run `python grid.py` once to record a fresh run.")
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    histories, context = state["histories"], state["context"]
    rows, cols = context["rows"], context["cols"]

    for (r, c) in cells:
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(
                f"cell [{r},{c}] is outside the {rows}x{cols} grid "
                f"(rows 0..{rows - 1}, cols 0..{cols - 1})")

    # 1) retrain only the requested cells, swap their flip-books in
    fresh = rerun_cells(context, cells, init_seed, device=device, lr=lr)
    for (r, c), history in fresh.items():
        histories[r][c] = history
        _render_cell(history, context, run_dir, r, c, fps)   # replace its dashboard

    # 2) refresh the overview so it reflects the replaced cells (good cells are
    #    redrawn from their cached, unchanged flip-books — identical output)
    render_grid(histories, context,
                out_gif=str(run_dir / "grid.gif"),
                out_png=str(run_dir / "grid.png"), fps=fps)

    _save_state(run_dir, histories, context)
    return {(r, c): fresh[(r, c)][-1] for (r, c) in fresh}
