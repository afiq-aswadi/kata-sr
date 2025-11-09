"""Reference solution for attribution patching."""

import torch
from transformer_lens import HookedTransformer


def attribution_patching(
    model: HookedTransformer,
    clean_text: str,
    corrupted_text: str,
    hook_point: str,
) -> torch.Tensor:
    """Measure impact of patching clean → corrupted for each position."""
    # Get baseline outputs
    clean_logits = model(clean_text)
    corrupted_logits = model(corrupted_text)

    # Cache clean activations
    _, cache = model.run_with_cache(clean_text)
    clean_activation = cache[hook_point]

    seq_len = clean_activation.shape[1]
    impacts = []

    # Patch each position and measure impact
    for pos in range(seq_len):
        def patch_hook(activation, hook):
            activation[:, pos, :] = clean_activation[:, pos, :]
            return activation

        patched_logits = model.run_with_hooks(
            corrupted_text, fwd_hooks=[(hook_point, patch_hook)]
        )

        # Measure impact as change in final logit
        impact = (patched_logits - corrupted_logits).abs().mean().item()
        impacts.append(impact)

    return torch.tensor(impacts)
