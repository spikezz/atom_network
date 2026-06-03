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

Uses **uv** with a project-local `.venv` (CPU-only PyTorch — the net is tiny).
Do **not** use conda.

```bash
uv sync                 # create .venv, install deps from pyproject.toml + uv.lock
uv run python main.py                                       # regression demo
uv run python main.py --task classification --epochs 2000   # classification demo
```

Outputs are written to `outputs/` as `<task>_training.gif` and `<task>_final.png`.

## Architecture

```
atom/
├── model.py       SimpleMLP — the 2-2-1 net; .trace() returns intermediate
│                  activations, .snapshot() returns weights/biases for drawing.
├── data.py        synthetic datasets (regression bowl, classification XOR) +
│                  true_surface() ground truth.
├── monitor.py     NetworkMonitor — PyTorch forward/backward hooks (read-only
│                  probes for activations + gradient norms).
├── train.py       training loop; records a frame dict every `snapshot_every`
│                  epochs. Returns (model, history, context).
└── visualize.py   render(history, context) -> GIF + PNG (matplotlib, Agg).
main.py            argparse CLI entry point.
```

Key design rule: **training records, visualization replays.** Keep those two
concerns separated — the same `history` can become an animation or a still.

## Conventions

- Dependencies live in `pyproject.toml`; torch is pinned to the CPU wheel index.
  After changing deps, run `uv lock` then `uv sync`.
- `.venv/`, `__pycache__/` are git-ignored. `outputs/` artifacts are kept (the
  README embeds `outputs/regression_final.png`).
- Code is written to be *read by a learner*: prefer clear names and comments
  that explain the "why" over cleverness.
