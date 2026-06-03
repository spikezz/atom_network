# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`atom_network` is an **educational visualization** of the smallest interesting
neural network: a 2 → 2 → 1 MLP (9 trainable parameters). The point is not the
model — it's making every weight, neuron, and gradient *visible* so a human can
watch the network learn. Training produces a per-frame "flip-book"; the
visualizer plays it back as an animated dashboard (network diagram + loss curve
+ learned-vs-truth function surfaces).

## Environment & Commands

Uses **uv** with a project-local `.venv`. torch is the **CUDA build** (cu128);
training auto-selects a device, preferring the **RTX 3070** by name (see
`atom/device.py`). Do **not** use conda.

```bash
uv sync                 # create .venv, install deps from pyproject.toml + uv.lock

# single network — the dashboard
uv run python main.py                                       # regression demo
uv run python main.py --task classification --epochs 2000   # classification demo

# capacity grid — many networks at once (→ wider, ↓ deeper)
# grid size/params come from config.yaml; CLI flags override it
uv run python grid.py                                       # uses config.yaml
uv run python grid.py --rows 3 --cols 3 --task classification

# force a device on either entry point
uv run python main.py --device cpu
```

Outputs go to `outputs/`: `<task>_training.gif` / `<task>_final.png` for the
single net, `grid_RxC_<task>.gif` / `.png` for the grid.

## Architecture

```
atom/
├── model.py       SimpleMLP (the 2-2-1 net) + MLP (general width/depth).
│                  Both expose .trace() (intermediate activations, on CPU) and
│                  .snapshot() (weights/biases, on CPU) for drawing.
├── data.py        synthetic datasets (regression bowl, classification XOR) +
│                  true_surface() ground truth.
├── monitor.py     NetworkMonitor — PyTorch forward/backward hooks (read-only
│                  probes for activations + gradient norms).
├── device.py      pick_device() — prefers the RTX 3070 by name, else any CUDA,
│                  else CPU; honours an explicit override.
├── config.py      load_config()/apply_overrides() — config.yaml driving grid.py
│                  (precedence: DEFAULTS < config.yaml < CLI flags).
├── train.py       single-net training loop; records a frame every
│                  `snapshot_every` epochs. Returns (model, history, context).
├── visualize.py   render(history, context) -> GIF + PNG (the 4-panel dashboard).
└── grid.py        train_grid()/render_grid() — a rows×cols grid of MLPs trained
│                  together; generic draw_net() handles any width/depth.
main.py            single-net CLI.
grid.py            capacity-grid CLI.
```

Key design rule: **training records, visualization replays.** Keep those two
concerns separated — the same `history` can become an animation or a still.
`.trace()`/`.snapshot()` always return CPU tensors so plotting works no matter
which device trained the model.

## Conventions

- Dependencies live in `pyproject.toml`; torch is pinned to the CPU wheel index.
  After changing deps, run `uv lock` then `uv sync`.
- `.venv/`, `__pycache__/` are git-ignored. `outputs/` artifacts are kept (the
  README embeds `outputs/regression_final.png`).
- Code is written to be *read by a learner*: prefer clear names and comments
  that explain the "why" over cleverness.
