"""Pick which device to train on, preferring a specific GPU by name.

This machine has several GPUs; we want the RTX 3070 first. If no CUDA GPU is
available we quietly fall back to CPU (which is actually faster for a single
tiny net — the GPU only earns its keep once the capacity grid grows).
"""

import torch

PREFERRED = ("3070",)  # substrings matched against GPU names, in priority order


def pick_device(prefer=PREFERRED, override: str | None = None) -> torch.device:
    """Return the device to use.

    override: "cpu", "cuda", or "cuda:N" forces a choice. Otherwise we look for
    a CUDA GPU whose name contains one of `prefer`, then any CUDA GPU, then CPU.
    """
    if override:
        return torch.device(override)
    if not torch.cuda.is_available():
        return torch.device("cpu")

    names = [(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
    for key in prefer:
        for i, name in names:
            if key in name:
                return torch.device(f"cuda:{i}")
    return torch.device("cuda:0")


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(device.index or 0)})"
    return str(device)
