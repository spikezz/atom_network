# atom_network 🔬

**A microscope for the smallest interesting neural network.**

Most neural networks are black boxes with millions of parameters. This one
has **nine**. It is small enough that you can put every weight, every neuron,
and every gradient on screen at once and *watch it learn* — frame by frame.

```
Input (2)  →  Hidden (2, ReLU)  →  Output (1)
```

That's it. Two inputs, two hidden neurons, one output. Nine numbers total
(6 weights + 3 biases). The whole project exists to make those nine numbers,
and what they do, visible to a human.

---

## What you get

Run it and you produce an animated dashboard with four synchronized panels:

| panel | what it shows |
|-------|---------------|
| **left** | the network itself — circles are neurons (coloured by their value), lines are weights (red = positive, blue = negative, thick = large). You are looking at the nine numbers being tuned in real time. |
| **top-right** | the **loss** falling as the network gets less wrong. |
| **bottom-middle** | the function the network represents **right now**. |
| **bottom-right** | the function it's **trying** to match (the ground truth). |

The story of learning is watching the bottom-middle morph into the
bottom-right, while the weights shift on the left and the loss drops on the
top-right — all at the same time.

![regression](outputs/regression_final.png)

---

## Quick start

This project uses [`uv`](https://docs.astral.sh/uv/) with a project-local
virtual environment (no global/conda install needed). PyTorch ships the CUDA
build; training **prefers the RTX 3070** automatically and falls back to CPU.
(A single tiny net is actually faster on CPU — the GPU earns its keep once the
capacity grid below grows. Force a device with `--device cpu` / `--device cuda:N`.)

```bash
uv sync                 # creates .venv and installs torch (CUDA), matplotlib, numpy

uv run python main.py                                   # regression: learn a smooth bowl
uv run python main.py --task classification --epochs 2000   # learn XOR (needs the hidden layer)
```

Outputs land in `outputs/` as both an animated `*.gif` and a final `*.png`.

### Options

```
--task {regression,classification}   what to learn        (default: regression)
--epochs N                           training epochs      (default: 1000)
--lr LR                              learning rate        (default: 0.01)
--snapshot-every N                   one frame per N epochs (default: 20)
--fps N                              gif frame rate       (default: 12)
--out DIR                            output directory     (default: outputs)
```

---

## The two tasks

- **regression** — learn `y = x₁² + x₂²`, a smooth bowl. With only two ReLU
  neurons the network can't fit it perfectly; watching *how it approximates*
  a curved surface with a few straight folds is itself the lesson.
- **classification** — learn the XOR-like rule "label 1 when x₁ and x₂ share a
  sign," a four-quadrant checkerboard. Two hidden neurons aren't quite enough
  to carve it cleanly: the network gets the dominant diagonal and stalls. That
  honest near-miss is the lesson — you are *seeing* the capacity limit of a
  9-parameter brain, not a bug.

---

## Is nine parameters enough? — the capacity grid

Short answer: no, not for the bowl. To *show* it, train a whole grid of
networks at once and watch them learn side by side. The grid is configured in
[`config.yaml`](config.yaml) — set the size there and just run it; any flag on
the command line overrides the file.

```yaml
# config.yaml
grid:
  rows: 2          # ↓ deeper  (each row adds a hidden layer)
  cols: 2          # → wider   (each col adds a neuron per hidden layer)
  base_width: 2    # the top-left cell
  base_depth: 1
training: { task: regression, epochs: 2000 }
```

```bash
uv run python grid.py                                   # uses config.yaml
uv run python grid.py --rows 3 --cols 4                 # CLI overrides the file
uv run python grid.py --config experiments/deep.yaml    # or point at another file
```

- **→ across columns**: each hidden layer gains one neuron (wider).
- **↓ down rows**: one more hidden layer is added (deeper).
- the input (2) and output (1) never change — they're fixed by the problem.

Every cell is a **full dashboard** — the same network diagram + loss curve +
learned-surface + truth you get for a single network — so no detail is lost;
only the colour scales are shared across the grid instead of repeated. The
2×2 regression grid makes the point bluntly — going from the corner 2-2-1
(9 params) to 2-3-3-1 (25 params) drops the loss by more than 10×:

![capacity grid](outputs/grid_2x2_regression.png)

### What a run produces

Each `grid.py` run gets its own timestamped folder under `outputs/`, named
`<timestamp>_<task>/`. Inside you get the whole grid **and** every cell broken
out on its own — each cell rendered as the exact same standalone dashboard a
single network gets, in its own subfolder:

```
outputs/
└── 20260604-000935_regression/        ← one folder per run (timestamp_task)
    ├── grid.gif / grid.png            ← the whole grid: overview + animation
    ├── config.yaml                    ← the exact settings that produced this run
    ├── run_state.pkl                  ← recorded run (so --rerun can patch cells)
    ├── r0c0_2-2-1/   2-2-1.gif|.png   ← every cell on its own (full dashboard)
    ├── r0c1_2-3-1/   2-3-1.gif|.png       r{row}c{col}_{arch}/
    ├── r1c0_2-2-2-1/ 2-2-2-1.gif|.png
    └── r1c1_2-3-3-1/ 2-3-3-1.gif|.png
```

The folder is **self-describing**: `config.yaml` is the *effective* config
(config.yaml + whatever you passed on the command line), so you can always read
off exactly what produced a run. (Run folders are git-ignored — regenerated on
every run.)

### Rerunning a cell that got stuck

Training is random: now and then a cell lands in a bad minimum (the deep,
narrow `2-2-2-1` is a repeat offender) and shows a much worse loss than its
neighbours. You don't have to retrain the whole grid — point `--rerun` at the
cell(s) by `[row, col]` and only those are retrained, **in place**:

```bash
uv run python grid.py --rerun "[1,0]" --lr 0.005      # same exact init, smaller lr
uv run python grid.py --rerun "[[1,0],[0,2]]" --lr 0.005   # several at once
uv run python grid.py --rerun "[1,0]" --seed 7        # a fresh, different init
```

- coordinates are `[row, col]` — row ↓ (depth), col → (width); the same axes
  the grid prints its losses on.
- only the listed cells are retrained and their dashboards replaced; every
  other cell's dashboard is left untouched. The grid overview is redrawn so it
  reflects the new result.
- **two init modes**, both keeping the data fixed:
  - **default (no `--seed`)** — the cell restarts from its **exact original
    weights**. Those starting numbers are saved with every run (inside
    `run_state.pkl`), so pairing the rerun with **`--lr`** asks one clean
    question: *with the very same starting point, does a different learning rate
    escape the minimum this cell fell into?* Nothing but the lr changes.
    (Re-seeding can't do this: in the grid all cells share one random stream and
    each draws from a different position, so the same seed value would give a
    *different* init — which is the whole point of keeping the actual weights.)
  - **`--seed N`** — start from a genuinely **fresh, different** init, for when
    you just want a stuck cell to get a new roll of the dice.
- **`--lr`** defaults to the saved run's rate; give it to change the rate.
- **which run?** by default it patches the *latest* `outputs/<timestamp>_<task>/`
  (the one closest to now); point at a specific one with
  `--run-dir outputs/20260604-000935_regression`. The replacement happens inside
  that folder.
- every cell's dashboard prints the `lr` and `seed` it was trained with, so you
  can always read off what produced it (in the default mode the seed shown is
  the grid's, i.e. the seed that originally made that init).

## How it's built

```
atom/
├── model.py       SimpleMLP — the 2-2-1 network, plus .trace() and .snapshot()
│                  so it can report its own internals for drawing.
├── data.py        the two synthetic datasets + the ground-truth surface.
├── monitor.py     NetworkMonitor — PyTorch forward/backward hooks that read
│                  activations and gradients without touching the model.
├── train.py       the training loop; every few epochs it freezes a "frame".
└── visualize.py   turns the sequence of frames into the GIF + PNG dashboard.
main.py            command-line entry point.
```

The design idea: **training produces a flip-book**, and **visualization plays
it back**. The two concerns stay cleanly separated, which is why the same
recorded history can become a smooth animation or a single crisp still.
