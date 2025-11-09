"""Patch residual stream and measure direct effect."""

import torch
from jaxtyping import Float
from transformer_lens import HookedTransformer


def patch_residual_stream(
    model: HookedTransformer,
    clean_text: str,
    corrupted_text: str,
    layer: int,
    position: int,
) -> Float[torch.Tensor, "d_model"]:
    """Get direct effect on residual stream at position.

    Patch the residual stream from clean into corrupted at a specific layer
    and position, then return the difference in the residual stream at that point.

    Args:
        model: HookedTransformer model
        clean_text: clean input text
        corrupted_text: corrupted input text
        layer: layer number to patch at
        position: token position to patch

    Returns:
        Delta in residual stream vector at (layer, position)
    """
    # TODO:
    # 1. Run corrupted with cache to get baseline residual stream
    # 2. Patch clean activation at (layer, position)
    # 3. Capture patched residual stream at same point
    # 4. Return difference (patched - baseline)
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
