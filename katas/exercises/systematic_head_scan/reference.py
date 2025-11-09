"""
Systematic Head Scan - Reference Implementation
"""

import torch
from transformer_lens import HookedTransformer
from typing import Dict


def compute_logit_diff(logits, answer_token, wrong_token):
    """Helper to compute logit difference."""
    return logits[0, -1, answer_token] - logits[0, -1, wrong_token]


def compute_patching_effect(
    patched_logits, clean_logits, corrupt_logits, answer_token, wrong_token
):
    """Helper to compute patching effect."""
    clean_diff = compute_logit_diff(clean_logits, answer_token, wrong_token)
    corrupt_diff = compute_logit_diff(corrupt_logits, answer_token, wrong_token)
    patched_diff = compute_logit_diff(patched_logits, answer_token, wrong_token)

    denominator = clean_diff - corrupt_diff
    if abs(denominator) < 1e-6:
        return 0.0

    return ((patched_diff - corrupt_diff) / denominator).item()


def scan_all_heads(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    answer_token: int,
    wrong_token: int
) -> torch.Tensor:
    """
    Patch all attention heads systematically and measure effects.
    """
    # Get baseline caches and logits
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)

    # Create results tensor
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    results = torch.zeros(n_layers, n_heads)

    # Scan all heads
    for layer in range(n_layers):
        for head in range(n_heads):
            # Patch this specific head
            hook_name = f"blocks.{layer}.attn.hook_z"

            def patch_hook(activation, hook):
                activation[:, :, head, :] = clean_cache[hook_name][:, :, head, :]
                return activation

            patched_logits = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[(hook_name, patch_hook)]
            )

            # Compute effect
            effect = compute_patching_effect(
                patched_logits, clean_logits, corrupt_logits,
                answer_token, wrong_token
            )

            results[layer, head] = effect

    return results
