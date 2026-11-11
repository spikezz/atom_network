# Restoring the local Python environment

The project uses `pyproject.toml` and the committed `uv.lock`. The local `.venv`
contains installed dependencies; project code, `config.yaml`, and `outputs/`
remain in the project when that environment is removed.

## Restore on Linux x86_64

Use uv and Python 3.11.9. The verified installer is uv 0.11.18. Python 3.11.9
remains installed separately from `.venv`; if it is missing on a new machine,
run `uv python install 3.11.9` first.

```bash
cd /home/spikezz/Project/atom_network
env -u UV_PROJECT_ENVIRONMENT uv sync --locked --no-cache --python 3.11.9
```

`--locked` prevents silently changing the recorded dependency versions.
`--no-cache` avoids retaining another persistent copy of the large CUDA wheels.
Use `pyproject.toml` and `uv.lock` from the same Git revision. Do not replace this
command with an unpinned `pip install torch` or delete the lockfile.

The lock selects PyTorch 2.11.0+cu128 and its CUDA 12.8 libraries. Packages are
retrieved from PyPI and the PyTorch CUDA 12.8 wheel index specified by the
project. Restoring requires network access and enough space for the 6.63 GiB
environment plus temporary download/extraction space. GPU use also requires a
compatible NVIDIA driver. No model checkpoint download is needed for this
small MLP project.

## Verify without using production GPUs

```bash
CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python - <<'PYTHON'
import torch
from atom.train import train
for task in ("regression", "classification"):
    model, history, context = train(task=task, epochs=3, n=16,
                                    snapshot_every=2, device="cpu")
    print(task, history[-1]["loss"])
print(torch.__version__, torch.version.cuda)
PYTHON
```

Then use the existing entry points, for example `uv run python main.py
--device cpu`. Running the full entry points can write to `outputs/`.

## Recovery record: 2026-11-11

The companion `environment-recovery.json` records the original Python,
installed package versions and dependency-file SHA-256 hashes. The only local
editable package is `atom-network`, whose source remains in this repository.
All 42 installed package versions matched the lockfile. All 6,367 installed
Python/JSON/YAML/TOML/PTH files checked against their installation RECORD
matched, with no missing or changed files.

A separate environment on the V disk was created from scratch with `uv sync
--locked --no-cache`, downloading all required packages successfully. All 42
package versions matched the original. Both regression and classification ran
three training steps on CPU, and GIF/PNG rendering passed. This verifies package
recovery and CPU execution; GPU execution was not tested during cleanup. The
recovery check used the existing Python 3.11.9 interpreter; downloading that
interpreter on a new host was not part of the check.

The temporary restoration environment was removed after successful verification.
The removed original `.venv` is reconstructed by the command above; there is no
archived binary environment to maintain.
