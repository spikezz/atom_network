"""atom_network — does more capacity help?

Train a grid of networks side by side and watch them learn the same target.
Across columns each hidden layer gains a neuron (wider); down rows a hidden
layer is added (deeper). The input (2) and output (1) stay fixed.

    python grid.py                          # 2x2 regression grid
    python grid.py --rows 3 --cols 3 --task classification
"""

import argparse
from pathlib import Path

from atom.grid import train_grid, render_grid
from atom.device import pick_device, describe


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=2, help="depth axis: number of rows")
    ap.add_argument("--cols", type=int, default=2, help="width axis: number of columns")
    ap.add_argument("--task", default="regression",
                    choices=["regression", "classification"])
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--snapshot-every", type=int, default=30)
    ap.add_argument("--device", default=None,
                    help="cpu | cuda | cuda:N (default: auto, prefers RTX 3070)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    device = pick_device(override=args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {describe(device)}")
    print(f"Training a {args.rows}x{args.cols} grid on '{args.task}' "
          f"for {args.epochs} epochs...")
    histories, context = train_grid(
        rows=args.rows, cols=args.cols, task=args.task, epochs=args.epochs,
        lr=args.lr, snapshot_every=args.snapshot_every, device=device,
    )

    # report the capacity-vs-fit story as plain numbers, too
    print("Final loss per cell (→ wider, ↓ deeper):")
    for r in range(args.rows):
        cells = []
        for c in range(args.cols):
            f = histories[r][c][-1]
            cells.append(f"{f['arch']}:{f['nparams']}p loss={f['loss']:.3f}")
        print("  " + "   |   ".join(cells))

    gif = out_dir / f"grid_{args.rows}x{args.cols}_{args.task}.gif"
    png = out_dir / f"grid_{args.rows}x{args.cols}_{args.task}.png"
    print("Rendering grid (the slow part)...")
    render_grid(histories, context, out_gif=str(gif), out_png=str(png), fps=args.fps)
    print(f"  GIF : {gif}")
    print(f"  PNG : {png}")


if __name__ == "__main__":
    main()
