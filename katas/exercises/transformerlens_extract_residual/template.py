"""Extract residual stream activations from TransformerLens cache."""

import torch
from transformer_lens import HookedTransformer


def extract_residual_stream(
    model: HookedTransformer, prompt: str, layer: int
) -> torch.Tensor:
    """Extract residual stream activations from a specific layer.

    Args:
        model: HookedTransformer model
        prompt: text prompt
        layer: layer number (0 to n_layers-1)

    Returns:
        residual stream tensor of shape (batch, seq_len, d_model)
    """
    # BLANK_START
    raise NotImplementedError(
        "Run model with cache and extract residual stream. "
        "Hint: cache key is f'blocks.{layer}.hook_resid_post'"
    )
    # BLANK_END
