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
from pathlib import Path

from atom.grid import train_grid, render_grid
from atom.device import pick_device, describe
from atom.config import load_config, apply_overrides


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
    ap.add_argument("--seed", type=int)
    ap.add_argument("--fps", type=int)
    ap.add_argument("--out")
    ap.add_argument("--device", help="cpu | cuda | cuda:N (default: auto, prefers RTX 3070)")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args)
    device = pick_device(override=cfg["device"])
    out_dir = Path(cfg["out"])
    out_dir.mkdir(parents=True, exist_ok=True)

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

    gif = out_dir / f"grid_{cfg['rows']}x{cfg['cols']}_{cfg['task']}.gif"
    png = out_dir / f"grid_{cfg['rows']}x{cfg['cols']}_{cfg['task']}.png"
    print("Rendering grid (the slow part)...")
    render_grid(histories, context, out_gif=str(gif), out_png=str(png), fps=cfg["fps"])
    print(f"  GIF : {gif}")
    print(f"  PNG : {png}")


if __name__ == "__main__":
    main()
