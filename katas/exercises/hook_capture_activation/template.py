"""Capture activations from TransformerLens models."""

import torch
from transformer_lens import HookedTransformer


def hook_capture_activation(
    model: HookedTransformer,
    text: str,
    hook_point: str,
) -> torch.Tensor:
    """Run model and capture activation at specific hook point.

    Args:
        model: HookedTransformer model
        text: input text
        hook_point: name of hook point (e.g., "blocks.0.attn.hook_z")

    Returns:
        Captured activation tensor
    """
    # TODO: Use model.run_with_cache to capture activation
    # Hint: logits, cache = model.run_with_cache(text)
    #       return cache[hook_point]
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
