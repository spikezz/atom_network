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

from .model import MLP
from .tasks import load_task
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
def _cell_frame(net, epoch, losses, grid, gx, gy, activation, lr, seed, init=None,
                init_kind=None):
    device = next(net.parameters()).device
    with torch.no_grad():
        pred = activation(net(grid).reshape(gx.shape))
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
        # the cell's epoch-0 weights — the *actual numbers* it drew from this
        # seed's stream. Same object in every frame (one slice, fixed), so it
        # pickles once; shown verbatim so two "seed 42" cells reveal their
        # different starting points. init_kind says where they came from
        # (grid slice / stream head / restored) so the caption can be honest.
        "init": init,
        "init_kind": init_kind,
    }


def train_grid(rows=2, cols=2, task="regression", epochs=1500, lr=0.01,
               n=1000, snapshot_every=30, seed=42, device=None,
               base_width=2, base_depth=1):
    """Train a rows×cols grid of MLPs on the same data; return (histories, ctx)."""
    if device is None:
        device = pick_device()
    task_obj = load_task(task)
    torch.manual_seed(seed)
    X, y = task_obj.make_data(n, seed)
    X, y = X.to(device), y.to(device)
    arch = make_architectures(rows, cols, base_width, base_depth)

    nets = [[MLP(hidden=arch[r][c]).to(device) for c in range(cols)] for r in range(rows)]
    # The pristine starting weights of every cell, captured before a single
    # training step. Saved with the run so a later --rerun can put a cell back to
    # its EXACT original init and change only the lr — a clean test of whether a
    # different rate escapes the minimum this init fell into. (Re-seeding can't do
    # this: in the grid all cells share one RNG stream and each draws from a
    # different position, so the same seed value gives a different init per cell.)
    init_weights = [[{k: v.detach().cpu().clone()
                      for k, v in nets[r][c].state_dict().items()}
                     for c in range(cols)] for r in range(rows)]
    # the same numbers in drawing form (per-layer W, b), to print on each cell
    init_snaps = [[nets[r][c].snapshot() for c in range(cols)] for r in range(rows)]
    opt = [[torch.optim.Adam(nets[r][c].parameters(), lr=lr) for c in range(cols)]
           for r in range(rows)]
    criterion = task_obj.make_criterion()

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
                                    grid, gx, gy, task_obj.activation, lr, seed,
                                    init=init_snaps[r][c], init_kind="grid"))

    context = {
        "task": task, "rows": rows, "cols": cols, "arch": arch,
        "gx": gx, "gy": gy, "true_surface": task_obj.true_surface(gx, gy),
        "grid_lim": GRID_LIM, "total_epochs": epochs, "device": str(device),
        # everything a later --rerun needs to retrain a single cell: same data +
        # same architecture grid, plus each cell's exact starting weights so the
        # retry can reproduce the original init (or take a fresh seed instead).
        "base_width": base_width, "base_depth": base_depth,
        "lr": lr, "n": n, "snapshot_every": snapshot_every, "seed": seed,
        "init_weights": init_weights,
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
    """Re-train just the listed (row, col) cells on the *same data* the grid used.
    Returns {(r, c): history}. Nothing here touches the other cells.

    Two init modes, chosen by ``init_seed``:

      * ``init_seed is None`` — restore the cell's EXACT original starting weights
        (saved with the run as ``context["init_weights"]``). Pair this with a new
        ``lr`` to ask the clean question "same start, does a different rate escape
        the minimum this cell fell into?" Re-seeding can't reproduce that init —
        in the grid every cell shares one RNG stream from a different position —
        which is why we keep the actual weights, not the seed number.
      * ``init_seed`` an int — a *fresh, different* start from that seed, so a
        stuck cell gets a genuinely new roll of the dice.

    ``lr`` overrides the learning rate for this retry; None reuses the rate the
    original grid trained with. Everything that would change the frame count
    (epochs, snapshot_every) stays locked to the run so the cell still lines up
    with its neighbours in the grid animation.
    """
    if device is None:
        device = pick_device()
    task = context["task"]
    task_obj = load_task(task)
    epochs = context["total_epochs"]
    lr = context["lr"] if lr is None else lr
    snapshot_every = context["snapshot_every"]
    arch_grid = context["arch"]
    gx, gy = context["gx"], context["gy"]
    init_weights = context.get("init_weights")

    # identical dataset to the original grid run (same seed) ...
    X, y = task_obj.make_data(context["n"], context["seed"])
    X, y = X.to(device), y.to(device)
    grid, _, _ = _input_grid()
    grid = grid.to(device)
    criterion = task_obj.make_criterion()

    out = {}
    for (r, c) in cells:
        hidden = tuple(arch_grid[r][c])
        if init_seed is None:
            # ... and the cell's EXACT original init, so only lr differs.
            if not init_weights or init_weights[r][c] is None:
                raise ValueError(
                    f"cell [{r},{c}] has no saved starting weights — this run was "
                    f"recorded before that was kept. Re-run `grid.py` once to "
                    f"capture them, or pass --seed N to start from a fresh init.")
            net = MLP(hidden=hidden).to(device)
            net.load_state_dict({k: v.to(device)
                                 for k, v in init_weights[r][c].items()})
            seed_used = context["seed"]        # the seed that produced this init
            init_kind = "restored"
        else:
            torch.manual_seed(init_seed)       # ... or a fresh, different start
            net = MLP(hidden=hidden).to(device)
            seed_used = init_seed
            init_kind = "head"                 # single net built first => stream head
        init = net.snapshot()                  # the actual numbers this retry used
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
                history.append(_cell_frame(net, epoch, losses, grid, gx, gy,
                                           task_obj.activation, lr, seed_used,
                                           init=init, init_kind=init_kind))
        out[(r, c)] = history
    return out


def rerun_run(run_dir, cells, init_seed, fps=12, device=None, lr=None):
    """Retrain the listed cells inside an existing run folder and replace them
    *in place*: each retried cell's own dashboard is re-rendered, and the grid
    overview is refreshed so it shows the new result — every other cell's
    standalone dashboard is left exactly as it was.

    ``init_seed`` None restores each cell's exact original init (saved with the
    run); an int starts from a fresh seed instead. ``lr`` overrides the learning
    rate for the retry (None = the run's own rate). See ``rerun_cells``.

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
