"""Tests for TransformerLens selective caching kata."""

import pytest
import torch
from transformer_lens import HookedTransformer


@pytest.fixture
try:
    from user_kata import cache_only_residual
except ImportError:
    from .reference import cache_only_residual


def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


def test_cache_only_residual_returns_cache(model):
    """Test that function returns a cache object."""

    cache = cache_only_residual(model, "Test")
    # Cache should be dict-like (support keys(), indexing, etc.)
    assert hasattr(cache, 'keys')
    assert hasattr(cache, '__getitem__')


def test_cache_contains_residual(model):
    """Test that cache contains residual activations."""

    cache = cache_only_residual(model, "Test")

    # Should have residual stream activations
    assert len(cache) > 0
    assert any("resid" in name for name in cache.keys())


def test_cache_only_has_residual(model):
    """Test that cache only contains residual (no attention or MLP)."""

    cache = cache_only_residual(model, "Test")

    # All keys should contain 'resid'
    assert all("resid" in name for name in cache.keys())

    # Should not have attention or MLP outputs
    assert not any("attn.hook_z" in name for name in cache.keys())
    assert not any("mlp_out" in name for name in cache.keys())


def test_cache_has_all_layers(model):
    """Test that cache has residual from all layers."""

    cache = cache_only_residual(model, "Test")
    n_layers = model.cfg.n_layers

    # Should have hook_resid_post from each layer
    for layer in range(n_layers):
        assert f"blocks.{layer}.hook_resid_post" in cache.keys()


def test_selective_cache_smaller_than_full(model):
    """Test that selective cache has fewer entries than full cache."""

    prompt = "Test"

    # Full cache
    tokens = model.to_tokens(prompt)
    logits, full_cache = model.run_with_cache(tokens)

    # Selective cache
    selective_cache = cache_only_residual(model, prompt)

    # Selective should have fewer entries
    assert len(selective_cache) < len(full_cache)
    assert len(selective_cache) > 0


def test_residual_values_correct(model):
    """Test that residual values match full cache."""

    prompt = "Test"

    # Full cache
    tokens = model.to_tokens(prompt)
    logits, full_cache = model.run_with_cache(tokens)

    # Selective cache
    selective_cache = cache_only_residual(model, prompt)

    # Values for residual should match
    for key in selective_cache.keys():
        assert key in full_cache
        assert torch.allclose(selective_cache[key], full_cache[key])


def test_different_prompts_different_caches(model):
    """Test that different prompts produce different caches."""

    cache1 = cache_only_residual(model, "Hello")
    cache2 = cache_only_residual(model, "Goodbye")

    # Same keys but different values
    key = "blocks.0.hook_resid_post"
    assert key in cache1 and key in cache2
    assert not torch.allclose(cache1[key], cache2[key])


def test_cache_has_embeddings(model):
    """Test that cache includes embedding residual stream."""

    cache = cache_only_residual(model, "Test")

    # Should have hook_embed and hook_pos_embed
    has_embed_related = any(
        "embed" in name or "resid_pre" in name for name in cache.keys()
    )
    assert has_embed_related


def test_cache_structure_valid(model):
    """Test that cached activations have valid structure."""

    cache = cache_only_residual(model, "Test input")

    # Each cached activation should be a tensor
    for name, activation in cache.items():
        assert isinstance(activation, torch.Tensor)
        assert torch.isfinite(activation).all()
