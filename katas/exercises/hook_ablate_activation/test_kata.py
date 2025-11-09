"""Tests for hook_ablate_activation kata."""

import torch
from transformer_lens import HookedTransformer

from framework import assert_shape

try:
    from user_kata import hook_ablate_activation
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    hook_ablate_activation = reference.hook_ablate_activation  # type: ignore


def test_output_shape():
    """Output should have correct logit shape."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    text = "Hello world"

    logits = hook_ablate_activation(model, text, "blocks.0.hook_resid_post", position=0)

    tokens = model.to_tokens(text)
    seq_len = tokens.shape[1]
    assert_shape(logits, (1, seq_len, model.cfg.d_vocab), "logits")


def test_ablation_changes_output():
    """Ablation should change model output."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    text = "The cat"

    # Normal output
    normal_logits = model(text)

    # Ablated output
    ablated_logits = hook_ablate_activation(
        model, text, "blocks.0.hook_resid_post", position=0
    )

    # Outputs should be different
    assert not torch.allclose(normal_logits, ablated_logits, atol=1e-3), \
        "ablation should change output"


def test_different_positions():
    """Should ablate different positions."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    text = "Hello world test"

    tokens = model.to_tokens(text)
    seq_len = tokens.shape[1]

    outputs = []
    for pos in range(seq_len):
        logits = hook_ablate_activation(
            model, text, "blocks.0.hook_resid_post", position=pos
        )
        outputs.append(logits)

    # All outputs should be different from each other
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            assert not torch.allclose(outputs[i], outputs[j], atol=1e-3), \
                f"ablating position {i} vs {j} should give different outputs"


def test_different_layers():
    """Should ablate different layers."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    text = "Test"

    normal_logits = model(text)

    for layer in [0, 5, 11]:
        ablated_logits = hook_ablate_activation(
            model, text, f"blocks.{layer}.hook_resid_post", position=0
        )
        assert not torch.allclose(normal_logits, ablated_logits, atol=1e-3), \
            f"ablating layer {layer} should change output"


def test_no_nans_or_infs():
    """Output should not contain NaNs or infinities."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    text = "Hello"

    logits = hook_ablate_activation(model, text, "blocks.0.hook_resid_post", position=0)

    assert not torch.isnan(logits).any(), "logits should not contain NaN"
    assert not torch.isinf(logits).any(), "logits should not contain inf"
