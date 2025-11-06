"""Cross-entropy loss kata - reference solution."""

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int


def cross_entropy_loss(
    logits: Float[torch.Tensor, "batch classes"],
    targets: Int[torch.Tensor, "batch"],
) -> Float[torch.Tensor, ""]:
    """Compute cross-entropy loss from logits and class indices."""
    return F.cross_entropy(logits, targets)


def binary_cross_entropy(
    predictions: Float[torch.Tensor, "batch"],
    targets: Float[torch.Tensor, "batch"],
) -> Float[torch.Tensor, ""]:
    """Compute binary cross-entropy loss."""
    eps = 1e-7
    predictions = torch.clamp(predictions, eps, 1 - eps)
    return -(targets * torch.log(predictions) + (1 - targets) * torch.log(1 - predictions)).mean()


def cross_entropy_with_label_smoothing(
    logits: Float[torch.Tensor, "batch classes"],
    targets: Int[torch.Tensor, "batch"],
    smoothing: float = 0.1,
) -> Float[torch.Tensor, ""]:
    """Cross-entropy with label smoothing regularization."""
    num_classes = logits.shape[-1]
    log_probs = F.log_softmax(logits, dim=-1)

    # Create smoothed targets
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)

    return (-true_dist * log_probs).sum(dim=-1).mean()
