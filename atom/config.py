"""Load the grid configuration from config.yaml.

Precedence, lowest to highest: built-in DEFAULTS  <  config.yaml  <  CLI flags.
The YAML is grouped (grid / training / render / device) for readability but we
flatten it into one plain dict the CLI can work with.
"""

from pathlib import Path
import yaml

DEFAULTS = {
    "rows": 2,
    "cols": 2,
    "base_width": 2,
    "base_depth": 1,
    "task": "regression",
    "epochs": 2000,
    "lr": 0.01,
    "snapshot_every": 40,
    "seed": 42,
    "fps": 12,
    "frame_repeat": 1,
    "out": "outputs",
    "device": None,
}

# which YAML section each key lives under (device sits at the top level)
_SECTIONS = {
    "rows": "grid", "cols": "grid", "base_width": "grid", "base_depth": "grid",
    "task": "training", "epochs": "training", "lr": "training",
    "snapshot_every": "training", "seed": "training",
    "fps": "render", "frame_repeat": "render", "out": "render",
    "device": None,
}


def load_config(path="config.yaml") -> dict:
    """Return a flat config dict (DEFAULTS overlaid with the YAML, if present)."""
    cfg = dict(DEFAULTS)
    p = Path(path)
    if not p.exists():
        return cfg

    raw = yaml.safe_load(p.read_text()) or {}
    for key, section in _SECTIONS.items():
        source = raw.get(section, {}) if section else raw
        if isinstance(source, dict) and key in source:
            cfg[key] = source[key]
        elif section is None and key in raw:
            cfg[key] = raw[key]
    return cfg


def apply_overrides(cfg: dict, args) -> dict:
    """Override config values with any CLI args that were actually provided
    (argparse leaves unprovided options as None)."""
    for key in DEFAULTS:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    return cfg


def save_config(cfg: dict, path) -> None:
    """Write the *effective* config used for a run (DEFAULTS + config.yaml + CLI
    overrides) back out in the same grouped layout, so a run folder is
    self-describing — you can read off exactly what produced it.

    This is a snapshot for the record only: --rerun reads run_state.pkl (which
    also carries the good cells' trained weights), not this file.
    """
    grouped = {"grid": {}, "training": {}, "render": {}}
    for key, section in _SECTIONS.items():
        if section is None:
            grouped[key] = cfg[key]          # device sits at the top level
        else:
            grouped[section][key] = cfg[key]
    header = (
        "# Effective configuration for THIS run.\n"
        "# = config.yaml overlaid with any CLI flags that were given —\n"
        "#   i.e. exactly what produced this folder.\n"
        "# For the record only; `grid.py --rerun` reads run_state.pkl, not this.\n"
    )
    Path(path).write_text(header + yaml.safe_dump(grouped, sort_keys=False))
