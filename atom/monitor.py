"""Looking inside the network with PyTorch hooks.

A hook is a callback PyTorch runs every time a layer fires. Forward hooks
see the activations a layer produced; backward hooks see the gradients
flowing back through it during learning. We never touch the model's own
code — we just attach probes and read what passes by.
"""

import warnings

import torch
import torch.nn as nn

# Benign: the first layer's backward hook fires even though the network input
# does not require grad. We only read grad_output, so this is harmless noise.
warnings.filterwarnings(
    "ignore", message="Full backward hook is firing when gradients are computed"
)


class NetworkMonitor:
    """Captures the most recent activations and gradient norms per layer."""

    def __init__(self):
        self.activations: dict[str, torch.Tensor] = {}
        self.grad_norms: dict[str, float] = {}
        self._handles: list = []

    def attach(self, model: nn.Module) -> "NetworkMonitor":
        """Probe every Linear layer in the model."""
        self.detach()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._handles.append(module.register_forward_hook(self._forward(name)))
                self._handles.append(module.register_full_backward_hook(self._backward(name)))
        return self

    def _forward(self, name: str):
        def hook(_module, _inp, out):
            self.activations[name] = out.detach().clone()
        return hook

    def _backward(self, name: str):
        def hook(_module, _grad_in, grad_out):
            g = grad_out[0]
            if g is not None:
                self.grad_norms[name] = g.detach().norm().item()
        return hook

    def detach(self) -> None:
        """Remove all hooks — always do this when monitoring is done."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
