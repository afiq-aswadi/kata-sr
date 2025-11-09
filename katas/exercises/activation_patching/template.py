"""Implement activation patching for causal interventions."""

import torch
from transformer_lens import HookedTransformer


def patch_residual_stream(
    model: HookedTransformer,
    clean_prompt: str,
    corrupted_prompt: str,
    layer: int,
    position: int,
) -> torch.Tensor:
    """Patch residual stream from clean run into corrupted run.

    Run model twice: once on clean prompt, once on corrupted. During the
    corrupted run, replace the residual stream at (layer, position) with
    the value from the clean run. Return the patched output logits.

    Args:
        model: HookedTransformer model
        clean_prompt: the "correct" input
        corrupted_prompt: the "incorrect" input to patch
        layer: which layer to patch
        position: which token position to patch

    Returns:
        logits from corrupted run with patched activation
    """
    # BLANK_START
    raise NotImplementedError(
        "1. Run clean prompt and cache activations (use remove_batch_dim=False)\n"
        "2. Extract clean residual at (layer, position): cache[...][0, position, :]\n"
        "3. Create hook that replaces corrupted residual with clean value\n"
        "4. Run corrupted prompt with hook and return logits\n"
        "Hint: use model.run_with_hooks() with fwd_hooks parameter"
    )
    # BLANK_END
