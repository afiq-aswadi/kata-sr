"""
Patching Effect Metric

Compute normalized patching effect to quantify causal importance of
model components.
"""

import torch


def compute_patching_effect(
    patched_logits: torch.Tensor,
    clean_logits: torch.Tensor,
    corrupt_logits: torch.Tensor,
    answer_token: int,
    wrong_token: int
) -> float:
    """
    Compute normalized patching effect using logit differences.

    The patching effect measures how much the intervention restored
    clean behavior, normalized so that:
    - 0.0 = no effect (still like corrupted)
    - 1.0 = full restoration (back to clean)
    - >1.0 = overcorrection (better than clean)

    Formula: (patched_diff - corrupt_diff) / (clean_diff - corrupt_diff)

    Args:
        patched_logits: Logits after patching [batch, seq_len, vocab]
        clean_logits: Logits from clean run [batch, seq_len, vocab]
        corrupt_logits: Logits from corrupted run [batch, seq_len, vocab]
        answer_token: Correct answer token ID
        wrong_token: Incorrect answer token ID

    Returns:
        Normalized patching effect as a float

    Example:
        >>> # After patching layer 5
        >>> effect = compute_patching_effect(
        ...     patched_logits, clean_logits, corrupt_logits,
        ...     mary_token, john_token
        ... )
        >>> print(f"Patching restored {effect:.1%} of clean behavior")
    """
    # BLANK_START
    raise NotImplementedError(
        "Compute logit_diff for all three runs, "
        "then normalize: (patched - corrupt) / (clean - corrupt)"
    )
    # BLANK_END
