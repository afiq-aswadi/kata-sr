"""Tests for TransformerLens activation caching kata."""

import pytest
import torch
from transformer_lens import HookedTransformer


@pytest.fixture
def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


def test_load_model():
    """Test that model loads correctly."""
    from template import load_model

    model = load_model("gpt2-small")
    assert isinstance(model, HookedTransformer)
    assert model.cfg.model_name == "gpt2-small"
    assert model.cfg.n_layers == 12
    assert model.cfg.d_model == 768


def test_run_with_full_cache(model):
    """Test running model with full cache."""
    from template import run_with_full_cache

    prompt = "The capital of France is"
    logits, cache = run_with_full_cache(model, prompt)

    # Logits should have shape (batch, seq_len, vocab_size)
    assert logits.dim() == 3
    assert logits.shape[-1] == model.cfg.d_vocab

    # Cache should contain many activation names
    assert len(cache) > 0
    assert any("resid" in name for name in cache.keys())
    assert any("attn" in name for name in cache.keys())
    assert any("mlp" in name for name in cache.keys())


def test_extract_residual_stream(model):
    """Test extracting residual stream from specific layer."""
    from template import extract_residual_stream

    prompt = "Hello world"
    layer = 0
    residual = extract_residual_stream(model, prompt, layer)

    # Should have shape (batch, seq_len, d_model)
    assert residual.dim() == 3
    assert residual.shape[0] == 1  # batch size
    assert residual.shape[2] == model.cfg.d_model


def test_extract_attention_output(model):
    """Test extracting attention output."""
    from template import extract_attention_output

    prompt = "Test input"
    layer = 0
    attn_out = extract_attention_output(model, prompt, layer)

    # Should have shape (batch, seq_len, n_heads, d_head)
    assert attn_out.dim() == 4
    assert attn_out.shape[0] == 1  # batch size
    assert attn_out.shape[2] == model.cfg.n_heads
    assert attn_out.shape[3] == model.cfg.d_head


def test_extract_mlp_output(model):
    """Test extracting MLP output."""
    from template import extract_mlp_output

    prompt = "Test input"
    layer = 0
    mlp_out = extract_mlp_output(model, prompt, layer)

    # Should have shape (batch, seq_len, d_model)
    assert mlp_out.dim() == 3
    assert mlp_out.shape[0] == 1  # batch size
    assert mlp_out.shape[2] == model.cfg.d_model


def test_extract_last_token_residual(model):
    """Test extracting last token position."""
    from template import extract_last_token_residual

    prompt = "The quick brown fox"
    layer = 0
    last_token = extract_last_token_residual(model, prompt, layer)

    # Should be 1D vector of size d_model
    assert last_token.dim() == 1
    assert last_token.shape[0] == model.cfg.d_model


def test_extract_specific_position(model):
    """Test extracting specific token position."""
    from template import extract_specific_position

    prompt = "Hello world test"
    layer = 0
    position = 1  # Second token
    token_vec = extract_specific_position(model, prompt, layer, position)

    # Should be 1D vector of size d_model
    assert token_vec.dim() == 1
    assert token_vec.shape[0] == model.cfg.d_model

    # Different positions should give different vectors
    pos0 = extract_specific_position(model, prompt, layer, 0)
    pos1 = extract_specific_position(model, prompt, layer, 1)
    assert not torch.allclose(pos0, pos1)


def test_run_with_selective_cache(model):
    """Test selective caching with filter function."""
    from template import run_with_selective_cache

    prompt = "Test"
    # Only cache residual stream activations
    filter_fn = lambda name: "resid" in name
    logits, cache = run_with_selective_cache(model, prompt, filter_fn)

    # Cache should only have resid activations
    assert len(cache) > 0
    assert all("resid" in name for name in cache.keys())
    assert not any("attn.hook_z" in name for name in cache.keys())
    assert not any("mlp_out" in name for name in cache.keys())


def test_cache_only_residual_stream(model):
    """Test caching only residual stream."""
    from template import cache_only_residual_stream

    prompt = "The capital of France"
    cache = cache_only_residual_stream(model, prompt)

    # Should only contain resid activations
    assert len(cache) > 0
    assert all("resid" in name for name in cache.keys())

    # Should have activations from all layers
    n_layers = model.cfg.n_layers
    resid_posts = [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]
    for name in resid_posts:
        assert name in cache.keys()


def test_compute_activation_stats(model):
    """Test computing statistics on activations."""
    from template import compute_activation_stats

    # Create a simple test tensor
    activations = torch.randn(10, 20, 768)
    stats = compute_activation_stats(activations)

    # Should have all required keys
    assert "mean" in stats
    assert "std" in stats
    assert "norm" in stats

    # All should be floats
    assert isinstance(stats["mean"], float)
    assert isinstance(stats["std"], float)
    assert isinstance(stats["norm"], float)

    # Values should be reasonable
    assert abs(stats["mean"]) < 1.0  # Should be close to 0 for randn
    assert 0.5 < stats["std"] < 1.5  # Should be close to 1 for randn
    assert stats["norm"] > 0  # Should be positive


def test_different_layers(model):
    """Test that different layers produce different activations."""
    from template import extract_residual_stream

    prompt = "Test"
    layer0 = extract_residual_stream(model, prompt, 0)
    layer1 = extract_residual_stream(model, prompt, 1)
    layer11 = extract_residual_stream(model, prompt, 11)

    # Different layers should have different values
    assert not torch.allclose(layer0, layer1)
    assert not torch.allclose(layer0, layer11)
    assert not torch.allclose(layer1, layer11)


def test_cache_memory_efficiency(model):
    """Test that selective caching uses less memory."""
    from template import run_with_full_cache, run_with_selective_cache

    prompt = "Test memory efficiency"

    # Full cache
    _, full_cache = run_with_full_cache(model, prompt)
    full_count = len(full_cache)

    # Selective cache (only resid)
    filter_fn = lambda name: "resid" in name
    _, selective_cache = run_with_selective_cache(model, prompt, filter_fn)
    selective_count = len(selective_cache)

    # Selective should have fewer entries
    assert selective_count < full_count
    assert selective_count > 0  # But not empty


def test_prompt_length_handling(model):
    """Test that different prompt lengths are handled correctly."""
    from template import extract_residual_stream

    short_prompt = "Hi"
    long_prompt = "The quick brown fox jumps over the lazy dog"

    short_resid = extract_residual_stream(model, short_prompt, 0)
    long_resid = extract_residual_stream(model, long_prompt, 0)

    # Should have different sequence lengths
    assert short_resid.shape[1] < long_resid.shape[1]

    # But same d_model dimension
    assert short_resid.shape[2] == long_resid.shape[2] == model.cfg.d_model
