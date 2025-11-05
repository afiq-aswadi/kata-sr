"""Tests for activation patching kata."""

import pytest
import torch
from transformer_lens import HookedTransformer


@pytest.fixture
def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


def test_patch_residual_stream(model):
    from template import patch_residual_stream

    clean = "The cat"
    corrupted = "The dog"

    clean_logits = model(clean)
    patched_logits = patch_residual_stream(model, clean, corrupted, layer=0, position=0)

    # Patching should change output
    assert not torch.allclose(clean_logits, patched_logits, atol=1e-3)

    # Clean up
    model.reset_hooks()


def test_patch_attention_head(model):
    from template import patch_attention_head

    clean = "The capital of France is"
    corrupted = "The capital of Germany is"

    clean_logits = model(clean)
    patched_logits = patch_attention_head(model, clean, corrupted, layer=5, head=0)

    # Patching should change output
    assert not torch.allclose(clean_logits, patched_logits, atol=1e-3)

    # Clean up
    model.reset_hooks()


def test_compute_patching_effect():
    from template import compute_patching_effect

    clean_logits = torch.zeros(1, 5, 100)
    corrupted_logits = torch.zeros(1, 5, 100)
    patched_logits = torch.zeros(1, 5, 100)

    # Set up scenario: clean=10, corrupted=0, patched=5
    target_token = 42
    clean_logits[0, -1, target_token] = 10.0
    corrupted_logits[0, -1, target_token] = 0.0
    patched_logits[0, -1, target_token] = 5.0

    effect = compute_patching_effect(clean_logits, corrupted_logits, patched_logits, target_token)

    # Should be 0.5 (halfway recovered)
    assert abs(effect - 0.5) < 0.01


def test_compute_patching_effect_full_recovery():
    from template import compute_patching_effect

    clean_logits = torch.zeros(1, 5, 100)
    corrupted_logits = torch.zeros(1, 5, 100)
    patched_logits = torch.zeros(1, 5, 100)

    target_token = 42
    clean_logits[0, -1, target_token] = 10.0
    corrupted_logits[0, -1, target_token] = 0.0
    patched_logits[0, -1, target_token] = 10.0  # Full recovery

    effect = compute_patching_effect(clean_logits, corrupted_logits, patched_logits, target_token)
    assert abs(effect - 1.0) < 0.01


def test_scan_all_heads(model):
    from template import scan_all_heads

    clean = "The cat sat"
    corrupted = "The dog sat"
    target_token = model.to_single_token(" on")

    results = scan_all_heads(model, clean, corrupted, target_token)

    # Should have shape (n_layers, n_heads)
    assert results.shape == (model.cfg.n_layers, model.cfg.n_heads)

    # Results should be real numbers
    assert not torch.isnan(results).any()

    # Clean up
    model.reset_hooks()
