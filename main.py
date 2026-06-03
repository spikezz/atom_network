"""atom_network — watch the smallest neural network learn.

Train a 2 -> 2 -> 1 MLP and render the whole process to an animated GIF plus
a final still image, so you can literally see the nine weights being tuned,
the loss fall, and the learned function grow into the target.

    python main.py                      # regression (a smooth bowl)
    python main.py --task classification --epochs 2000
"""

import argparse
from pathlib import Path

from atom.train import train
from atom.visualize import render


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="regression",
                    choices=["regression", "classification"],
                    help="what the network should learn")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--snapshot-every", type=int, default=20,
                    help="capture one animation frame every N epochs")
    ap.add_argument("--out", default="outputs", help="output directory")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training {args.task} for {args.epochs} epochs (lr={args.lr})...")
    model, history, context = train(
        task=args.task, epochs=args.epochs, lr=args.lr,
        snapshot_every=args.snapshot_every,
    )
    print(f"Done. Final loss = {history[-1]['loss']:.6f} "
          f"over {len(history)} captured frames.")

    gif = out_dir / f"{args.task}_training.gif"
    png = out_dir / f"{args.task}_final.png"
    print("Rendering visualization (this is the slow part)...")
    render(history, context, out_gif=str(gif), out_png=str(png), fps=args.fps)
    print(f"  GIF : {gif}")
    print(f"  PNG : {png}")


if __name__ == "__main__":
    main()
