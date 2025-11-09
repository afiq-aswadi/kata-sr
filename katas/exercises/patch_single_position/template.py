"""Patch a single token position between clean and corrupted runs."""

import torch
from transformer_lens import HookedTransformer


def patch_position(
    model: HookedTransformer,
    clean_text: str,
    corrupted_text: str,
    hook_point: str,
    position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patch single token position, return (clean_output, patched_output).

    Args:
        model: HookedTransformer model
        clean_text: clean input text
        corrupted_text: corrupted input text
        hook_point: hook point to patch at (e.g., "blocks.0.hook_resid_post")
        position: token position to patch

    Returns:
        Tuple of (clean_logits, patched_logits)
        - clean_logits: output from clean run
        - patched_logits: output from corrupted run with clean activation patched at position
    """
    # TODO:
    # 1. Run clean text to get clean logits and cache
    # 2. Create hook that patches clean activation at position into corrupted run
    # 3. Run corrupted text with hook to get patched logits
    # 4. Return (clean_logits, patched_logits)
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
