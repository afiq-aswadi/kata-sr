"""
Logit Difference Metric

Implement the logit difference metric used in mechanistic interpretability
to measure model performance on tasks like Indirect Object Identification (IOI).
"""

import torch


def compute_logit_diff(
    logits: torch.Tensor,
    answer_token: int,
    wrong_token: int,
    position: int = -1
) -> torch.Tensor:
    """
    Compute the difference between logits for answer and wrong tokens.

    This metric is commonly used to measure how strongly a model predicts
    the correct answer vs an incorrect alternative.

    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        answer_token: Token ID for the correct answer
        wrong_token: Token ID for the incorrect answer
        position: Which position to check (default: -1 for last token)

    Returns:
        Scalar tensor: logits[answer] - logits[wrong]

    Example:
        >>> logits = model("When Mary and John went to the store, John gave a drink to")
        >>> mary_token = model.to_single_token(" Mary")
        >>> john_token = model.to_single_token(" John")
        >>> diff = compute_logit_diff(logits, mary_token, john_token)
        >>> # Positive diff means model favors Mary (correct for IOI)
    """
    # BLANK_START
    raise NotImplementedError(
        "Index logits at [batch=0, position, token_id] and return difference"
    )
    # BLANK_END
