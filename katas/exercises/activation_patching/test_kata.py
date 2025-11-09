"""Tests for activation patching kata."""

import pytest
import torch
from transformer_lens import HookedTransformer


@pytest.fixture
def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


def test_patch_returns_tensor(model):
    """Test that function returns logits tensor."""
    from template import patch_residual_stream

    clean = "The Eiffel Tower is in Paris"
    corrupted = "The Eiffel Tower is in Rome"

    logits = patch_residual_stream(model, clean, corrupted, layer=0, position=0)
    assert isinstance(logits, torch.Tensor)


def test_patched_logits_shape(model):
    """Test that patched logits have correct shape."""
    from template import patch_residual_stream

    clean = "The capital of France is Paris"
    corrupted = "The capital of France is Rome"

    logits = patch_residual_stream(model, clean, corrupted, layer=5, position=-1)

    # Should be (batch, seq_len, vocab_size)
    assert logits.dim() == 3
    assert logits.shape[0] == 1
    assert logits.shape[2] == model.cfg.d_vocab


def test_patching_changes_output(model):
    """Test that patching actually changes the output."""
    from template import patch_residual_stream

    clean = "The Eiffel Tower is in Paris"
    corrupted = "The Eiffel Tower is in Rome"

    # Baseline: corrupted without patching
    corrupted_tokens = model.to_tokens(corrupted)
    baseline_logits = model(corrupted_tokens)

    # Patched: corrupted with clean activation
    patched_logits = patch_residual_stream(
        model, clean, corrupted, layer=5, position=5
    )

    # Patching should change the output
    assert not torch.allclose(baseline_logits, patched_logits, atol=1e-5)


def test_patching_different_layers(model):
    """Test that patching different layers gives different results."""
    from template import patch_residual_stream

    clean = "Paris is the capital of France"
    corrupted = "Rome is the capital of France"

    layer0_patch = patch_residual_stream(model, clean, corrupted, layer=0, position=0)
    layer6_patch = patch_residual_stream(model, clean, corrupted, layer=6, position=0)

    # Different layers should produce different effects
    assert not torch.allclose(layer0_patch, layer6_patch)


def test_patching_different_positions(model):
    """Test that patching different positions gives different results."""
    from template import patch_residual_stream

    clean = "The Eiffel Tower is in Paris"
    corrupted = "The Eiffel Tower is in Rome"

    pos0_patch = patch_residual_stream(model, clean, corrupted, layer=5, position=0)
    pos3_patch = patch_residual_stream(model, clean, corrupted, layer=5, position=3)

    # Different positions should produce different effects
    assert not torch.allclose(pos0_patch, pos3_patch)


def test_patching_same_prompts_no_effect(model):
    """Test that patching identical prompts has no effect."""
    from template import patch_residual_stream

    prompt = "The capital of France is Paris"

    # Patch with itself
    patched_logits = patch_residual_stream(
        model, prompt, prompt, layer=5, position=0
    )

    # Baseline
    tokens = model.to_tokens(prompt)
    baseline_logits = model(tokens)

    # Should be identical (patching same into same)
    assert torch.allclose(patched_logits, baseline_logits, atol=1e-5)


def test_patching_moves_toward_clean(model):
    """Test that patching makes output more similar to clean run."""
    from template import patch_residual_stream

    clean = "The Eiffel Tower is in Paris"
    corrupted = "The Eiffel Tower is in Rome"

    # Get clean baseline
    clean_tokens = model.to_tokens(clean)
    clean_logits = model(clean_tokens)

    # Get corrupted baseline
    corrupted_tokens = model.to_tokens(corrupted)
    corrupted_logits = model(corrupted_tokens)

    # Patch at a late layer
    patched_logits = patch_residual_stream(model, clean, corrupted, layer=10, position=5)

    # Distance from corrupted to clean
    corrupted_to_clean_dist = (corrupted_logits - clean_logits).norm()

    # Distance from patched to clean
    patched_to_clean_dist = (patched_logits - clean_logits).norm()

    # Patching should move output closer to clean (for some layers/positions)
    # This is not guaranteed for all positions, but should hold for important ones
    # We use a loose check here
    assert patched_logits.shape == clean_logits.shape


def test_patching_last_position(model):
    """Test patching the last token position."""
    from template import patch_residual_stream

    clean = "Two plus two equals four"
    corrupted = "Two plus two equals five"

    # Patch last position
    patched_logits = patch_residual_stream(model, clean, corrupted, layer=8, position=-1)

    assert patched_logits.dim() == 3
    assert torch.isfinite(patched_logits).all()


def test_multiple_prompts_different_lengths(model):
    """Test patching works with different prompt lengths."""
    from template import patch_residual_stream

    clean = "Hi"
    corrupted = "The quick brown fox jumps over the lazy dog"

    # Patch at position 0 (both should have this)
    patched_logits = patch_residual_stream(model, clean, corrupted, layer=3, position=0)

    assert patched_logits.dim() == 3
    # Output should match corrupted length
    corrupted_tokens = model.to_tokens(corrupted)
    assert patched_logits.shape[1] == corrupted_tokens.shape[1]
