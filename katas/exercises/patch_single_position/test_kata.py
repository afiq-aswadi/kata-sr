"""Tests for patch_single_position kata."""

import torch
from transformer_lens import HookedTransformer

from framework import assert_shape

try:
    from user_kata import patch_position
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    patch_position = reference.patch_position  # type: ignore


def test_output_shapes():
    """Both outputs should have correct logit shape."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean_text = "The cat sat"
    corrupted_text = "The dog ran"

    clean_logits, patched_logits = patch_position(
        model, clean_text, corrupted_text, "blocks.0.hook_resid_post", position=0
    )

    tokens = model.to_tokens(clean_text)
    seq_len = tokens.shape[1]

    assert_shape(clean_logits, (1, seq_len, model.cfg.d_vocab), "clean logits")
    assert_shape(patched_logits, (1, seq_len, model.cfg.d_vocab), "patched logits")


def test_patching_changes_corrupted_output():
    """Patched output should differ from pure corrupted output."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean_text = "The cat sat"
    corrupted_text = "The dog ran"

    corrupted_logits = model(corrupted_text)
    _, patched_logits = patch_position(
        model, clean_text, corrupted_text, "blocks.0.hook_resid_post", position=1
    )

    assert not torch.allclose(corrupted_logits, patched_logits, atol=1e-3), \
        "patching should change corrupted output"


def test_different_positions():
    """Should patch different positions."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean_text = "Hello world test"
    corrupted_text = "Goodbye moon trial"

    tokens = model.to_tokens(clean_text)
    seq_len = tokens.shape[1]

    patched_outputs = []
    for pos in range(min(seq_len, 3)):
        _, patched = patch_position(
            model, clean_text, corrupted_text, "blocks.0.hook_resid_post", position=pos
        )
        patched_outputs.append(patched)

    # Different positions should give different outputs
    for i in range(len(patched_outputs)):
        for j in range(i + 1, len(patched_outputs)):
            assert not torch.allclose(patched_outputs[i], patched_outputs[j], atol=1e-3), \
                f"patching position {i} vs {j} should give different outputs"


def test_clean_output_unchanged():
    """Clean logits should match direct model call."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean_text = "Test text"
    corrupted_text = "Other text"

    expected_clean = model(clean_text)
    clean_logits, _ = patch_position(
        model, clean_text, corrupted_text, "blocks.0.hook_resid_post", position=0
    )

    assert torch.allclose(clean_logits, expected_clean, atol=1e-5), \
        "clean logits should match direct model output"


def test_different_layers():
    """Should patch at different layers."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean_text = "Test"
    corrupted_text = "Other"

    for layer in [0, 5, 11]:
        clean, patched = patch_position(
            model, clean_text, corrupted_text, f"blocks.{layer}.hook_resid_post", position=0
        )
        tokens = model.to_tokens(clean_text)
        seq_len = tokens.shape[1]
        assert_shape(clean, (1, seq_len, model.cfg.d_vocab), f"clean layer {layer}")
        assert_shape(patched, (1, seq_len, model.cfg.d_vocab), f"patched layer {layer}")


def test_no_nans_or_infs():
    """Outputs should not contain NaNs or infinities."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean_text = "Hello"
    corrupted_text = "World"

    clean_logits, patched_logits = patch_position(
        model, clean_text, corrupted_text, "blocks.0.hook_resid_post", position=0
    )

    assert not torch.isnan(clean_logits).any(), "clean logits should not contain NaN"
    assert not torch.isinf(clean_logits).any(), "clean logits should not contain inf"
    assert not torch.isnan(patched_logits).any(), "patched logits should not contain NaN"
    assert not torch.isinf(patched_logits).any(), "patched logits should not contain inf"
