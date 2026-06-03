"""atom_network — does more capacity help?

Train a grid of networks side by side and watch them learn the same target.
Across columns each hidden layer gains a neuron (wider); down rows a hidden
layer is added (deeper). The input (2) and output (1) stay fixed.

The grid is configured in **config.yaml** — just edit it and run:

    python grid.py

Any setting can still be overridden on the command line (it wins over the file):

    python grid.py --rows 3 --cols 4 --task classification
"""

import argparse
import ast
from datetime import datetime
from pathlib import Path

from atom.grid import train_grid, render_run, rerun_run
from atom.device import pick_device, describe
from atom.config import load_config, apply_overrides, save_config


def parse_cells(s):
    """Parse a --rerun string into a list of (row, col) cells.

    Accepts a single cell   "[0,2]"
    or several cells         "[[0,2],[1,1]]"  /  "[0,2],[1,1]"
    Coordinates are [row, col]  (row ↓ depth axis, col → width axis).
    """
    val = ast.literal_eval(s)
    # one cell: a bare [row, col]
    if (isinstance(val, (list, tuple)) and len(val) == 2
            and all(isinstance(x, int) for x in val)):
        return [(int(val[0]), int(val[1]))]
    # many cells: an iterable of [row, col] pairs
    cells = []
    for p in val:
        if not (isinstance(p, (list, tuple)) and len(p) == 2):
            raise ValueError(f"bad cell {p!r}; expected [row, col]")
        cells.append((int(p[0]), int(p[1])))
    if not cells:
        raise ValueError("no cells given")
    return cells


def latest_run_dir(out_dir, task):
    """The most recent outputs/<timestamp>_<task>/ that has a saved run state."""
    runs = sorted(p for p in out_dir.glob(f"*_{task}")
                  if p.is_dir() and (p / "run_state.pkl").exists())
    return runs[-1] if runs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml",
                    help="path to the YAML config (default: config.yaml)")
    # all default to None so we can tell "not given" from "given"; None => use config
    ap.add_argument("--rows", type=int, help="↓ depth axis: number of rows")
    ap.add_argument("--cols", type=int, help="→ width axis: number of columns")
    ap.add_argument("--base-width", dest="base_width", type=int,
                    help="neurons per hidden layer in the top-left cell")
    ap.add_argument("--base-depth", dest="base_depth", type=int,
                    help="hidden layers in the top-left cell")
    ap.add_argument("--task", choices=["regression", "classification"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--snapshot-every", dest="snapshot_every", type=int)
    ap.add_argument("--seed", type=int,
                    help="grid seed; with --rerun, start the retried cell(s) from "
                         "a fresh init seeded with this. Omit it to instead restore "
                         "each cell's EXACT original init (study lr in isolation).")
    ap.add_argument("--fps", type=int)
    ap.add_argument("--out")
    ap.add_argument("--device", help="cpu | cuda | cuda:N (default: auto, prefers RTX 3070)")
    # rerun only the cells that trained badly, leaving the good ones untouched
    ap.add_argument("--rerun", metavar="CELLS",
                    help='retrain only these grid cell(s) in an existing run, then '
                         'replace them in place. Coords are [row,col] (row ↓ depth, '
                         'col → width), e.g. --rerun "[0,2]" or --rerun "[[0,2],[1,1]]". '
                         'By default the cell restarts from its EXACT original init, '
                         'so pairing it with --lr isolates the effect of the learning '
                         'rate; pass --seed N to start from a fresh init instead.')
    ap.add_argument("--run-dir", dest="run_dir",
                    help="which run folder --rerun targets (default: the latest "
                         "outputs/<timestamp>_<task>/ with a saved run state)")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args)
    device = pick_device(override=cfg["device"])
    out_dir = Path(cfg["out"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- rerun mode: surgically retrain just the failed cells, then return ---
    if args.rerun is not None:
        try:
            cells = parse_cells(args.rerun)
        except (ValueError, SyntaxError) as e:
            ap.error(f"could not parse --rerun {args.rerun!r}: {e}. "
                     f'Use [row,col], e.g. --rerun "[1,0]" or "[[1,0],[0,2]]".')
        if args.run_dir:
            run_dir = Path(args.run_dir)
        else:
            run_dir = latest_run_dir(out_dir, cfg["task"])
            if run_dir is None:
                ap.error(f"no run folder with a saved state found under {out_dir}/ "
                         f"for task '{cfg['task']}'. Run `python grid.py` first, "
                         f"or point at one with --run-dir.")
        # no --seed => restore each cell's exact original init (init_seed=None);
        # --seed N => a fresh, different start from that seed.
        init_seed = args.seed
        seed_note = (f"fresh init, seed {init_seed}" if init_seed is not None
                     else "restoring each cell's exact original init")
        lr_note = f"lr {args.lr} (override)" if args.lr is not None else "lr from the saved run"
        print(f"Rerun: {run_dir}  |  device: {describe(device)}")
        print(f"  cells {cells}  |  {seed_note}  |  {lr_note}")
        try:
            last = rerun_run(run_dir, cells, init_seed, fps=cfg["fps"],
                             device=device, lr=args.lr)
        except (ValueError, FileNotFoundError) as e:
            ap.error(str(e))
        print("New loss for the retrained cell(s):")
        for (r, c), f in last.items():
            print(f"  [{r},{c}] {f['arch']}:{f['nparams']}p  loss={f['loss']:.3f}  "
                  f"(lr {f['lr']:g}, seed {f['seed']})")
        print(f"Replaced {len(last)} cell(s) in place; the other cells were left "
              f"untouched and the grid overview was refreshed.\n  {run_dir}/")
        return

    print(f"Config: {args.config}  |  device: {describe(device)}")
    print(f"Training a {cfg['rows']}x{cfg['cols']} grid on '{cfg['task']}' "
          f"for {cfg['epochs']} epochs "
          f"(base {cfg['base_width']}w x {cfg['base_depth']}d)...")
    histories, context = train_grid(
        rows=cfg["rows"], cols=cfg["cols"], task=cfg["task"], epochs=cfg["epochs"],
        lr=cfg["lr"], snapshot_every=cfg["snapshot_every"], seed=cfg["seed"],
        device=device, base_width=cfg["base_width"], base_depth=cfg["base_depth"],
    )

    print("Final loss per cell (→ wider, ↓ deeper):")
    for r in range(cfg["rows"]):
        cells = [f"{histories[r][c][-1]['arch']}:{histories[r][c][-1]['nparams']}p "
                 f"loss={histories[r][c][-1]['loss']:.3f}" for c in range(cfg["cols"])]
        print("  " + "   |   ".join(cells))

    # one folder per run: outputs/<timestamp>_<task>/
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = out_dir / f"{ts}_{cfg['task']}"
    print(f"Rendering into {run_dir}/ (the slow part) ...")
    print("  · grid.gif / grid.png        — the whole grid")
    print("  · r{r}c{c}_<arch>/<arch>.*   — every cell on its own (gif + png)")
    print("  · config.yaml                — the exact settings that produced this run")
    _, cell_dirs = render_run(histories, context, run_dir, fps=cfg["fps"])
    save_config(cfg, run_dir / "config.yaml")   # make the run folder self-describing
    print(f"Done. {1 + len(cell_dirs)} dashboards "
          f"(1 grid + {len(cell_dirs)} cells) under:\n  {run_dir}/")
    print(f"Rerun a stuck cell with:\n"
          f"  uv run python grid.py --rerun \"[r,c]\" --run-dir {run_dir}")


if __name__ == "__main__":
    main()
