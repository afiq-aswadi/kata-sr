"""Layer normalization implementation kata."""

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

        # TODO: create learnable scale (gamma) and shift (beta) parameters
        # BLANK_START
        pass
        # BLANK_END

    def forward(
        self, x: Float[torch.Tensor, "... features"]
    ) -> Float[torch.Tensor, "... features"]:
        """Apply layer normalization.

        Args:
            x: input tensor with features as last dimension

        Returns:
            normalized tensor with same shape as input
        """
        # TODO: compute mean and variance, normalize, then scale and shift
        # BLANK_START
        pass
        # BLANK_END
