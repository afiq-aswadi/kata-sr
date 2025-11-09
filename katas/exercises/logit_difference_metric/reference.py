"""
Logit Difference Metric - Reference Implementation
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
    """
    answer_logit = logits[0, position, answer_token]
    wrong_logit = logits[0, position, wrong_token]
    return answer_logit - wrong_logit
