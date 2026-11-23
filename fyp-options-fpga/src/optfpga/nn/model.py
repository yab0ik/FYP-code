"""PyTorch neural-network model."""

from typing import Sequence

import torch
from torch import nn


class MLP(nn.Module):
    """Plain fully connected network for call-price approximation."""

    def __init__(
        self,
        input_size: int = 4,
        hidden_sizes: Sequence[int] | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 32]
        if activation not in ("relu", "tanh"):
            raise ValueError("activation must be 'relu' or 'tanh'")

        activation_class = nn.ReLU if activation == "relu" else nn.Tanh
        layers: list[nn.Module] = []
        previous = input_size
        for size in hidden_sizes:
            if size <= 0:
                raise ValueError("hidden sizes must be positive")
            layers.extend((nn.Linear(previous, size), activation_class()))
            previous = size
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
