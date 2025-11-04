"""MLP implementation kata."""

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

        # TODO: create layers with Linear and ReLU
        # BLANK_START
        pass
        # BLANK_END

    def forward(
        self, x: Float[torch.Tensor, "batch input_dim"]
    ) -> Float[torch.Tensor, "batch output_dim"]:
        """Forward pass.

        Args:
            x: input tensor

        Returns:
            output tensor
        """
        # TODO: apply layers
        # BLANK_START
        pass
        # BLANK_END
