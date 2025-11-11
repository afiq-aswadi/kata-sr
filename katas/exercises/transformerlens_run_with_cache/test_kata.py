"""Tests for TransformerLens run_with_cache kata."""

import pytest
import torch
from transformer_lens import HookedTransformer


@pytest.fixture
try:
    from user_kata import run_with_cache
except ImportError:
    from .reference import run_with_cache


def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


def test_run_with_cache_returns_tuple(model):
    """Test that function returns (logits, cache) tuple."""

    prompt = "Hello world"
    result = run_with_cache(model, prompt)

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_logits_shape(model):
    """Test that logits have correct shape."""

    prompt = "The quick brown fox"
    logits, cache = run_with_cache(model, prompt)

    # Logits should be (batch, seq_len, vocab_size)
    assert logits.dim() == 3
    assert logits.shape[0] == 1  # batch size
    assert logits.shape[2] == model.cfg.d_vocab


def test_cache_contains_activations(model):
    """Test that cache contains activation names."""

    prompt = "Test"
    logits, cache = run_with_cache(model, prompt)

    # Cache should be dict-like with activation names
    assert hasattr(cache, 'keys')
    assert len(cache) > 0


def test_cache_has_residual_stream(model):
    """Test that cache contains residual stream activations."""

    prompt = "Test"
    logits, cache = run_with_cache(model, prompt)

    # Should have residual stream from multiple layers
    assert any("resid" in name for name in cache.keys())
    assert "blocks.0.hook_resid_post" in cache.keys()


def test_cache_has_attention(model):
    """Test that cache contains attention activations."""

    prompt = "Test"
    logits, cache = run_with_cache(model, prompt)

    # Should have attention activations
    assert any("attn" in name for name in cache.keys())


def test_cache_has_mlp(model):
    """Test that cache contains MLP activations."""

    prompt = "Test"
    logits, cache = run_with_cache(model, prompt)

    # Should have MLP activations
    assert any("mlp" in name for name in cache.keys())


def test_different_prompts_different_outputs(model):
    """Test that different prompts produce different results."""

    logits1, _ = run_with_cache(model, "Hello")
    logits2, _ = run_with_cache(model, "Goodbye")

    # Different prompts should produce different logits
    assert not torch.allclose(logits1, logits2)


def test_prompt_length_affects_sequence_length(model):
    """Test that longer prompts create longer sequences."""

    short_prompt = "Hi"
    long_prompt = "The quick brown fox jumps over the lazy dog"

    logits_short, _ = run_with_cache(model, short_prompt)
    logits_long, _ = run_with_cache(model, long_prompt)

    # Longer prompt should have more tokens
    assert logits_short.shape[1] < logits_long.shape[1]
