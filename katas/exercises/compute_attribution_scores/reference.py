"""Reference solution for computing attribution scores."""

import torch
from jaxtyping import Float
from transformer_lens import HookedTransformer


def compute_attribution_scores(
    model: HookedTransformer,
    clean_text: str,
    corrupted_text: str,
    metric_fn: callable,
) -> dict[str, Float[torch.Tensor, "..."]]:
    """Score all layers/positions for attribution."""
    # Get baselines
    clean_logits = model(clean_text)
    corrupted_logits = model(corrupted_text)

    clean_score = metric_fn(clean_logits)
    corrupted_score = metric_fn(corrupted_logits)

    # Cache clean activations
    _, clean_cache = model.run_with_cache(clean_text)

    results = {}

    # Iterate over layers
    for layer in range(model.cfg.n_layers):
        hook_point = f"blocks.{layer}.hook_resid_post"
        clean_activation = clean_cache[hook_point]
        seq_len = clean_activation.shape[1]

        layer_scores = []

        # Patch each position
        for pos in range(seq_len):
            def patch_hook(activation, hook):
                activation[:, pos, :] = clean_activation[:, pos, :]
                return activation

            patched_logits = model.run_with_hooks(
                corrupted_text, fwd_hooks=[(hook_point, patch_hook)]
            )
            patched_score = metric_fn(patched_logits)

            # Attribution score
            if abs(clean_score - corrupted_score) > 1e-8:
                score = (patched_score - corrupted_score) / (clean_score - corrupted_score)
            else:
                score = 0.0

            layer_scores.append(score)

        results[f"layer_{layer}"] = torch.tensor(layer_scores)

    return results
