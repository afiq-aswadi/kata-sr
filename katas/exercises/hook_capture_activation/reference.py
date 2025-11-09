"""Reference solution for capturing activations."""

import torch
from transformer_lens import HookedTransformer


def hook_capture_activation(
    model: HookedTransformer,
    text: str,
    hook_point: str,
) -> torch.Tensor:
    """Run model and capture activation at specific hook point."""
    _, cache = model.run_with_cache(text)
    return cache[hook_point]
