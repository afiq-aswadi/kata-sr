"""Tests for patch_residual_stream kata."""

import torch
from transformer_lens import HookedTransformer

from framework import assert_shape

try:
    from user_kata import patch_residual_stream
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    patch_residual_stream = reference.patch_residual_stream  # type: ignore


def test_output_shape():
    """Delta should be a d_model vector."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean_text = "The cat sat"
    corrupted_text = "The dog ran"

    delta = patch_residual_stream(model, clean_text, corrupted_text, layer=0, position=1)

    assert_shape(delta, (model.cfg.d_model,), "residual stream delta")


def test_delta_nonzero():
    """Delta should be non-zero when clean and corrupted differ."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean_text = "The cat"
    corrupted_text = "The dog"

    delta = patch_residual_stream(model, clean_text, corrupted_text, layer=0, position=1)

    # Should have some non-zero effect
    assert delta.abs().sum() > 1e-6, "patching should have measurable effect"


def test_different_positions():
    """Different positions should give different deltas."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean_text = "Hello world test"
    corrupted_text = "Goodbye moon trial"

    tokens = model.to_tokens(clean_text)
    seq_len = tokens.shape[1]

    deltas = []
    for pos in range(min(seq_len, 3)):
        delta = patch_residual_stream(
            model, clean_text, corrupted_text, layer=0, position=pos
        )
        deltas.append(delta)

    # Different positions should give different deltas
    for i in range(len(deltas)):
        for j in range(i + 1, len(deltas)):
            assert not torch.allclose(deltas[i], deltas[j], atol=1e-3), \
                f"position {i} vs {j} should give different deltas"


def test_different_layers():
    """Different layers should give different deltas."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean_text = "Test"
    corrupted_text = "Other"

    deltas = []
    for layer in [0, 5, 11]:
        delta = patch_residual_stream(
            model, clean_text, corrupted_text, layer=layer, position=0
        )
        deltas.append(delta)
        assert_shape(delta, (model.cfg.d_model,), f"layer {layer}")

    # Different layers should give different deltas
    for i in range(len(deltas)):
        for j in range(i + 1, len(deltas)):
            # Allow some similarity but should generally differ
            if not torch.allclose(deltas[i], deltas[j], atol=1e-2):
                break
    else:
        assert False, "at least some layers should give different deltas"


def test_no_nans_or_infs():
    """Delta should not contain NaNs or infinities."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean_text = "Hello"
    corrupted_text = "World"

    delta = patch_residual_stream(model, clean_text, corrupted_text, layer=0, position=0)

    assert not torch.isnan(delta).any(), "delta should not contain NaN"
    assert not torch.isinf(delta).any(), "delta should not contain inf"
