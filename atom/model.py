"""The smallest interesting neural network.

Two inputs -> two hidden neurons (ReLU) -> one output. That is exactly
nine trainable numbers (4 + 2 weights, 2 + 1 biases), few enough that we
can draw every single one of them on screen at the same time. That is the
whole point of this project: a network small enough to *see*.
"""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Input(2) -> Hidden(2, ReLU) -> Output(1)."""

    def __init__(self, input_size: int = 2, hidden_size: int = 2, output_size: int = 1):
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.act(self.hidden(x)))

    @torch.no_grad()
    def trace(self, x: torch.Tensor) -> dict:
        """Run a forward pass and hand back every intermediate tensor (on CPU,
        ready for plotting), so a human can watch a value flow through."""
        hidden_pre = self.hidden(x)
        hidden_post = self.act(hidden_pre)
        output = self.output(hidden_post)
        return {
            "input": x.cpu(),
            "hidden_pre": hidden_pre.cpu(),
            "hidden_post": hidden_post.cpu(),
            "output": output.cpu(),
        }

    @torch.no_grad()
    def snapshot(self) -> dict:
        """Current weights and biases as plain detached tensors, for drawing.

        Shapes: W1 (hidden, input), b1 (hidden,), W2 (output, hidden), b2 (output,).
        """
        return {
            "W1": self.hidden.weight.detach().cpu().clone(),
            "b1": self.hidden.bias.detach().cpu().clone(),
            "W2": self.output.weight.detach().cpu().clone(),
            "b2": self.output.bias.detach().cpu().clone(),
        }


class MLP(nn.Module):
    """A general MLP of any width and depth, so we can compare capacities.

    `hidden` is a tuple of hidden-layer widths, e.g. (2,) is one hidden layer
    of two neurons, (3, 3) is two hidden layers of three. Every hidden layer
    uses ReLU; the output layer is linear. This is the same family as
    SimpleMLP — SimpleMLP is exactly MLP(hidden=(2,)) — just parameterised so
    a grid can grow it one neuron / one layer at a time.
    """

    def __init__(self, in_dim: int = 2, hidden=(2,), out_dim: int = 1):
        super().__init__()
        self.sizes = [in_dim, *hidden, out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(self.sizes[i], self.sizes[i + 1])
            for i in range(len(self.sizes) - 1)
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lin in enumerate(self.layers):
            x = lin(x)
            if i < len(self.layers) - 1:  # no activation after the output layer
                x = self.act(x)
        return x

    @torch.no_grad()
    def trace(self, x: torch.Tensor) -> list:
        """Activations at every node-layer (on CPU): [input, hidden1, ..., output]."""
        acts = [x.cpu()]
        h = x
        for i, lin in enumerate(self.layers):
            h = lin(h)
            if i < len(self.layers) - 1:
                h = self.act(h)
            acts.append(h.cpu())
        return acts

    @torch.no_grad()
    def snapshot(self) -> list:
        """Per-layer (weight, bias) tensors on CPU, ordered input→output."""
        return [(lin.weight.detach().cpu().clone(), lin.bias.detach().cpu().clone())
                for lin in self.layers]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def arch_str(self) -> str:
        return "-".join(str(s) for s in self.sizes)

