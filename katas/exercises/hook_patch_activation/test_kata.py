"""Tests for hook_patch_activation kata."""

import torch
from transformer_lens import HookedTransformer

from framework import assert_shape

try:
    from user_kata import hook_patch_activation
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    hook_patch_activation = reference.hook_patch_activation  # type: ignore


def test_output_shape():
    """Output should have correct logit shape."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    target_text = "The cat"
    source_text = "The dog"

    logits = hook_patch_activation(
        model, target_text, source_text, "blocks.0.hook_resid_post", position=0
    )

    tokens = model.to_tokens(target_text)
    seq_len = tokens.shape[1]
    assert_shape(logits, (1, seq_len, model.cfg.d_vocab), "logits")


def test_patching_changes_output():
    """Patching should change model output."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    target_text = "The cat sat"
    source_text = "The dog ran"

    # Normal target output
    normal_logits = model(target_text)

    # Patched output
    patched_logits = hook_patch_activation(
        model, target_text, source_text, "blocks.0.hook_resid_post", position=1
    )

    # Outputs should be different
    assert not torch.allclose(normal_logits, patched_logits, atol=1e-3), \
        "patching should change output"


def test_different_positions():
    """Should patch different positions."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    target_text = "Hello world test"
    source_text = "Goodbye moon trial"

    tokens = model.to_tokens(target_text)
    seq_len = tokens.shape[1]

    outputs = []
    for pos in range(min(seq_len, 3)):  # Test first 3 positions
        logits = hook_patch_activation(
            model, target_text, source_text, "blocks.0.hook_resid_post", position=pos
        )
        outputs.append(logits)

    # All outputs should be different from each other
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            assert not torch.allclose(outputs[i], outputs[j], atol=1e-3), \
                f"patching position {i} vs {j} should give different outputs"


def test_different_layers():
    """Should patch different layers."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    target_text = "Test text"
    source_text = "Other text"

    normal_logits = model(target_text)

    for layer in [0, 5, 11]:
        patched_logits = hook_patch_activation(
            model, target_text, source_text, f"blocks.{layer}.hook_resid_post", position=0
        )
        assert not torch.allclose(normal_logits, patched_logits, atol=1e-3), \
            f"patching layer {layer} should change output"


def test_no_nans_or_infs():
    """Output should not contain NaNs or infinities."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    target_text = "Hello"
    source_text = "World"

    logits = hook_patch_activation(
        model, target_text, source_text, "blocks.0.hook_resid_post", position=0
    )

    assert not torch.isnan(logits).any(), "logits should not contain NaN"
    assert not torch.isinf(logits).any(), "logits should not contain inf"
