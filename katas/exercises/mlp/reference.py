"""Reference implementation for MLP kata."""

import torch
import torch.nn as nn
from jaxtyping import Float


class MLP(nn.Module):
    """Multi-layer perceptron with ReLU activations.

    Args:
        input_dim: input feature dimension
        hidden_dims: list of hidden layer dimensions
        output_dim: output dimension
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(
        self, x: Float[torch.Tensor, "batch input_dim"]
    ) -> Float[torch.Tensor, "batch output_dim"]:
        """Forward pass.

        Args:
            x: input tensor

        Returns:
            output tensor
        """
        return self.layers(x)
