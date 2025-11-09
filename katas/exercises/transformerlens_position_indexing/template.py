"""Extract position-specific activations from TransformerLens cache."""

import torch
from transformer_lens import HookedTransformer


def extract_position(
    model: HookedTransformer, prompt: str, layer: int, position: int
) -> torch.Tensor:
    """Extract residual stream at a specific token position.

    Args:
        model: HookedTransformer model
        prompt: text prompt
        layer: layer number
        position: token position (0-indexed, or -1 for last token)

    Returns:
        residual stream vector at that position, shape (d_model,)
    """
    # BLANK_START
    raise NotImplementedError(
        "Extract residual stream and index to specific position. "
        "Hint: residual[0, position, :] gets position from batch"
    )
    # BLANK_END
