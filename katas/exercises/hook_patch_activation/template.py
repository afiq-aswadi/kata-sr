"""Patch activations from one run into another."""

import torch
from transformer_lens import HookedTransformer


def hook_patch_activation(
    model: HookedTransformer,
    target_text: str,
    source_text: str,
    hook_point: str,
    position: int,
) -> torch.Tensor:
    """Patch activation from source_text into target_text run.

    Args:
        model: HookedTransformer model
        target_text: text to run with patched activation
        source_text: text to get activation from
        hook_point: name of hook point
        position: token position to patch

    Returns:
        Model output logits with patched activation
    """
    # TODO:
    # 1. Run source_text with cache to get source activation
    # 2. Define hook that replaces target activation with source at position
    # 3. Run target_text with hook
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
