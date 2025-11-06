"""Cross-entropy loss kata."""

import torch
from jaxtyping import Float, Int


def cross_entropy_loss(
    logits: Float[torch.Tensor, "batch classes"],
    targets: Int[torch.Tensor, "batch"],
) -> Float[torch.Tensor, ""]:
    """Compute cross-entropy loss from logits and class indices.

    Args:
        logits: unnormalized predictions (batch, num_classes)
        targets: true class indices (batch,)

    Returns:
        scalar loss (mean over batch)
    """
    # TODO: implement using log_softmax + nll_loss or from scratch
    # BLANK_START
    pass
    # BLANK_END


def binary_cross_entropy(
    predictions: Float[torch.Tensor, "batch"],
    targets: Float[torch.Tensor, "batch"],
) -> Float[torch.Tensor, ""]:
    """Compute binary cross-entropy loss.

    Args:
        predictions: predicted probabilities in [0, 1]
        targets: true labels (0 or 1)

    Returns:
        scalar loss (mean over batch)
    """
    # TODO: implement -[t*log(p) + (1-t)*log(1-p)]
    # BLANK_START
    pass
    # BLANK_END


def cross_entropy_with_label_smoothing(
    logits: Float[torch.Tensor, "batch classes"],
    targets: Int[torch.Tensor, "batch"],
    smoothing: float = 0.1,
) -> Float[torch.Tensor, ""]:
    """Cross-entropy with label smoothing regularization.

    Args:
        logits: unnormalized predictions
        targets: true class indices
        smoothing: label smoothing factor

    Returns:
        scalar loss
    """
    # TODO: smooth targets: (1-smoothing)*one_hot + smoothing/num_classes
    # BLANK_START
    pass
    # BLANK_END
