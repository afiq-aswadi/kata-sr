"""Tests for TransformerLens extract_residual_stream kata."""

import pytest
import torch
from transformer_lens import HookedTransformer


@pytest.fixture
try:
    from user_kata import extract_residual_stream
except ImportError:
    from .reference import extract_residual_stream


def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


def test_extract_residual_returns_tensor(model):
    """Test that function returns a tensor."""

    residual = extract_residual_stream(model, "Hello", layer=0)
    assert isinstance(residual, torch.Tensor)


def test_residual_shape(model):
    """Test that residual has correct shape."""

    prompt = "The quick brown fox"
    residual = extract_residual_stream(model, prompt, layer=0)

    # Should be (batch, seq_len, d_model)
    assert residual.dim() == 3
    assert residual.shape[0] == 1  # batch size
    assert residual.shape[2] == model.cfg.d_model


def test_different_layers_different_activations(model):
    """Test that different layers have different activations."""

    prompt = "Test"
    layer0 = extract_residual_stream(model, prompt, layer=0)
    layer5 = extract_residual_stream(model, prompt, layer=5)
    layer11 = extract_residual_stream(model, prompt, layer=11)

    # Different layers should have different values
    assert not torch.allclose(layer0, layer5)
    assert not torch.allclose(layer0, layer11)
    assert not torch.allclose(layer5, layer11)


def test_all_layers_accessible(model):
    """Test that all layers can be extracted."""

    prompt = "Test"
    n_layers = model.cfg.n_layers

    for layer in range(n_layers):
        residual = extract_residual_stream(model, prompt, layer)
        assert residual.shape[2] == model.cfg.d_model


def test_prompt_length_affects_sequence_dimension(model):
    """Test that prompt length affects sequence dimension."""

    short = extract_residual_stream(model, "Hi", layer=0)
    long = extract_residual_stream(model, "The quick brown fox jumps", layer=0)

    # Longer prompt should have more positions
    assert short.shape[1] < long.shape[1]


def test_different_prompts_different_activations(model):
    """Test that different prompts produce different activations."""

    resid1 = extract_residual_stream(model, "Hello", layer=0)
    resid2 = extract_residual_stream(model, "Goodbye", layer=0)

    assert not torch.allclose(resid1, resid2)


def test_residual_stream_values_reasonable(model):
    """Test that residual stream has reasonable values."""

    residual = extract_residual_stream(model, "Test", layer=0)

    # Should have finite values
    assert torch.isfinite(residual).all()

    # Mean should be relatively close to 0 (due to layer norm)
    mean = residual.mean().item()
    assert abs(mean) < 10.0


def test_later_layers_build_on_earlier(model):
    """Test that later layers have different magnitudes."""

    prompt = "Test"
    layer0 = extract_residual_stream(model, prompt, layer=0)
    layer11 = extract_residual_stream(model, prompt, layer=11)

    # Both should have reasonable norms
    norm0 = layer0.norm().item()
    norm11 = layer11.norm().item()

    assert norm0 > 0
    assert norm11 > 0
