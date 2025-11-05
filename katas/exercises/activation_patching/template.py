"""Activation patching kata."""

import torch
from transformer_lens import HookedTransformer


def patch_residual_stream(
    model: HookedTransformer,
    clean_text: str,
    corrupted_text: str,
    layer: int,
    position: int,
) -> torch.Tensor:
    """Patch residual stream from corrupted into clean run.

    Args:
        model: HookedTransformer model
        clean_text: clean input text
        corrupted_text: corrupted input text
        layer: layer to patch at
        position: position to patch

    Returns:
        logits with patched activation
    """
    # TODO:
    # 1. Run corrupted text with cache
    # 2. Create hook that replaces activation at (layer, position) with corrupted
    # 3. Run clean text with hook
    # BLANK_START
    pass
    # BLANK_END


def patch_attention_head(
    model: HookedTransformer,
    clean_text: str,
    corrupted_text: str,
    layer: int,
    head: int,
) -> torch.Tensor:
    """Patch attention head output from corrupted into clean run.

    Args:
        model: HookedTransformer model
        clean_text: clean input
        corrupted_text: corrupted input
        layer: layer number
        head: head number

    Returns:
        logits with patched head
    """
    # TODO: patch specific attention head (hook_z)
    # BLANK_START
    pass
    # BLANK_END


def compute_patching_effect(
    clean_logits: torch.Tensor,
    corrupted_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    target_token: int,
) -> float:
    """Compute how much patching recovered clean behavior.

    Args:
        clean_logits: logits from clean run
        corrupted_logits: logits from corrupted run
        patched_logits: logits from patched run
        target_token: target token ID to measure

    Returns:
        recovery percentage (0 = no recovery, 1 = full recovery)
    """
    # TODO: compute (patched - corrupted) / (clean - corrupted) for target token
    # BLANK_START
    pass
    # BLANK_END


def scan_all_heads(
    model: HookedTransformer,
    clean_text: str,
    corrupted_text: str,
    target_token: int,
) -> torch.Tensor:
    """Scan all attention heads to find most important ones.

    Args:
        model: HookedTransformer model
        clean_text: clean input
        corrupted_text: corrupted input
        target_token: target token ID

    Returns:
        tensor of shape (n_layers, n_heads) with patching effects
    """
    # TODO: iterate over all layers and heads, compute patching effect
    # BLANK_START
    pass
    # BLANK_END
