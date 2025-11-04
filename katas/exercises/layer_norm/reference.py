"""Reference implementation for layer normalization kata."""

import torch
import torch.nn as nn
from jaxtyping import Float


class LayerNorm(nn.Module):
    """Layer normalization module.

    Normalizes across the last dimension (features).

    Args:
        normalized_shape: size of the feature dimension
        eps: small constant for numerical stability
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

        # learnable scale and shift parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(
        self, x: Float[torch.Tensor, "... features"]
    ) -> Float[torch.Tensor, "... features"]:
        """Apply layer normalization.

        Args:
            x: input tensor with features as last dimension

        Returns:
            normalized tensor with same shape as input
        """
        # compute mean and variance over feature dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        # normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # scale and shift
        return self.gamma * x_norm + self.beta
