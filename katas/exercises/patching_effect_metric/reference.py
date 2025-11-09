"""
Patching Effect Metric - Reference Implementation
"""

import torch


def compute_logit_diff(logits, answer_token, wrong_token):
    """Helper to compute logit difference."""
    return logits[0, -1, answer_token] - logits[0, -1, wrong_token]


def compute_patching_effect(
    patched_logits: torch.Tensor,
    clean_logits: torch.Tensor,
    corrupt_logits: torch.Tensor,
    answer_token: int,
    wrong_token: int
) -> float:
    """
    Compute normalized patching effect using logit differences.
    """
    clean_diff = compute_logit_diff(clean_logits, answer_token, wrong_token)
    corrupt_diff = compute_logit_diff(corrupt_logits, answer_token, wrong_token)
    patched_diff = compute_logit_diff(patched_logits, answer_token, wrong_token)

    # Avoid division by zero
    denominator = clean_diff - corrupt_diff
    if abs(denominator) < 1e-6:
        return 0.0

    effect = (patched_diff - corrupt_diff) / denominator
    return effect.item()
