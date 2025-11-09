"""Tests for attribution_patching kata."""

import torch
from transformer_lens import HookedTransformer

from framework import assert_shape

try:
    from user_kata import attribution_patching
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    attribution_patching = reference.attribution_patching  # type: ignore


def test_output_shape():
    """Output should have shape (seq_len,)."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean_text = "The cat sat"
    corrupted_text = "The dog ran"

    impacts = attribution_patching(
        model, clean_text, corrupted_text, "blocks.0.hook_resid_post"
    )

    tokens = model.to_tokens(clean_text)
    seq_len = tokens.shape[1]
    assert_shape(impacts, (seq_len,), "attribution impacts")


def test_impacts_non_negative():
    """Impact scores should be non-negative."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean_text = "Hello world"
    corrupted_text = "Goodbye moon"

    impacts = attribution_patching(
        model, clean_text, corrupted_text, "blocks.0.hook_resid_post"
    )

    assert (impacts >= 0).all(), "impacts should be non-negative"


def test_different_texts():
    """Should work with different text pairs."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")

    pairs = [
        ("The cat", "The dog"),
        ("Hello world", "Goodbye moon"),
        ("I am happy", "I am sad"),
    ]

    for clean, corrupted in pairs:
        impacts = attribution_patching(
            model, clean, corrupted, "blocks.0.hook_resid_post"
        )
        tokens = model.to_tokens(clean)
        seq_len = tokens.shape[1]
        assert_shape(impacts, (seq_len,), f"{clean} vs {corrupted}")


def test_different_layers():
    """Should work with different layers."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean_text = "Test"
    corrupted_text = "Other"

    for layer in [0, 5, 11]:
        impacts = attribution_patching(
            model, clean_text, corrupted_text, f"blocks.{layer}.hook_resid_post"
        )
        tokens = model.to_tokens(clean_text)
        seq_len = tokens.shape[1]
        assert_shape(impacts, (seq_len,), f"layer {layer}")


def test_some_positions_have_impact():
    """At least some positions should have non-zero impact."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean_text = "The cat sat on the mat"
    corrupted_text = "The dog ran in the park"

    impacts = attribution_patching(
        model, clean_text, corrupted_text, "blocks.5.hook_resid_post"
    )

    # At least one position should have meaningful impact
    assert (impacts > 1e-6).any(), "some positions should have impact"


def test_no_nans_or_infs():
    """Impact scores should not contain NaNs or infinities."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean_text = "Hello"
    corrupted_text = "World"

    impacts = attribution_patching(
        model, clean_text, corrupted_text, "blocks.0.hook_resid_post"
    )

    assert not torch.isnan(impacts).any(), "impacts should not contain NaN"
    assert not torch.isinf(impacts).any(), "impacts should not contain inf"
