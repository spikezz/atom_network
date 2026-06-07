"""Task registry — one module per learning problem, discovered by filename.

The `task:` value in config.yaml (or `--task`) names a module in this package;
`load_task("regression")` imports `atom/tasks/regression.py` and returns its
`TASK`. Adding a task means adding a file here and nothing else: `available_
tasks()` scans the folder, so the CLIs and config validation pick it up for free.
"""

import importlib
import pkgutil
from pathlib import Path

from .base import Task

# modules in this package that are infrastructure, not tasks
_NON_TASKS = {"base"}


def available_tasks() -> list:
    """Sorted names of every task module in this package (one .py per task)."""
    pkg_dir = Path(__file__).parent
    return sorted(m.name for m in pkgutil.iter_modules([str(pkg_dir)])
                  if m.name not in _NON_TASKS)


def load_task(name: str) -> Task:
    """Return the `Task` defined by `atom/tasks/<name>.py` (its `TASK`)."""
    if name not in available_tasks():
        raise ValueError(
            f"unknown task {name!r}; available: {available_tasks()}")
    module = importlib.import_module(f".{name}", __package__)
    return module.TASK
