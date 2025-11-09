"""
Systematic Head Scan

Systematically patch all attention heads to discover causal circuits.
"""

import torch
from transformer_lens import HookedTransformer
from typing import Dict


def scan_all_heads(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    answer_token: int,
    wrong_token: int
) -> torch.Tensor:
    """
    Patch all attention heads systematically and measure effects.

    This function creates a heatmap of shape [n_layers, n_heads] where
    each entry shows the patching effect of that specific head.

    Args:
        model: TransformerLens model
        clean_tokens: Clean prompt tokens
        corrupt_tokens: Corrupted prompt tokens
        answer_token: Correct answer token ID
        wrong_token: Incorrect answer token ID

    Returns:
        Tensor of shape [n_layers, n_heads] with patching effects

    Example:
        >>> results = scan_all_heads(
        ...     model, clean_tokens, corrupt_tokens, mary_token, john_token
        ... )
        >>> # Visualize which heads matter
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(results)
        >>> plt.xlabel("Head")
        >>> plt.ylabel("Layer")
    """
    # BLANK_START
    raise NotImplementedError(
        "1. Run both clean and corrupt to get caches and baseline logits\n"
        "2. Create results tensor of zeros [n_layers, n_heads]\n"
        "3. Loop over all layers and heads\n"
        "4. For each: patch that head and compute patching effect\n"
        "5. Store effect in results[layer, head]"
    )
    # BLANK_END
