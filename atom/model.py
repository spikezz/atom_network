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
        """Run a forward pass and hand back every intermediate tensor, so a
        human (or a plot) can watch a value flow through the network."""
        hidden_pre = self.hidden(x)
        hidden_post = self.act(hidden_pre)
        output = self.output(hidden_post)
        return {
            "input": x,
            "hidden_pre": hidden_pre,
            "hidden_post": hidden_post,
            "output": output,
        }

    @torch.no_grad()
    def snapshot(self) -> dict:
        """Current weights and biases as plain detached tensors, for drawing.

        Shapes: W1 (hidden, input), b1 (hidden,), W2 (output, hidden), b2 (output,).
        """
        return {
            "W1": self.hidden.weight.detach().clone(),
            "b1": self.hidden.bias.detach().clone(),
            "W2": self.output.weight.detach().clone(),
            "b2": self.output.bias.detach().clone(),
        }
