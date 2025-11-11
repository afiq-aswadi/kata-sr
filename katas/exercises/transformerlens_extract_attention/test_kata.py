"""Tests for TransformerLens extract_attention_output kata."""

import pytest
import torch
from transformer_lens import HookedTransformer


@pytest.fixture
try:
    from user_kata import extract_attention_output
except ImportError:
    from .reference import extract_attention_output


def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


def test_extract_attention_returns_tensor(model):
    """Test that function returns a tensor."""

    attn = extract_attention_output(model, "Hello", layer=0)
    assert isinstance(attn, torch.Tensor)


def test_attention_output_shape(model):
    """Test that attention output has correct shape."""

    prompt = "Test input"
    attn = extract_attention_output(model, prompt, layer=0)

    # Should be (batch, seq_len, n_heads, d_head)
    assert attn.dim() == 4
    assert attn.shape[0] == 1  # batch size
    assert attn.shape[2] == model.cfg.n_heads
    assert attn.shape[3] == model.cfg.d_head


def test_different_layers_different_outputs(model):
    """Test that different layers have different attention outputs."""

    prompt = "Test"
    layer0 = extract_attention_output(model, prompt, layer=0)
    layer5 = extract_attention_output(model, prompt, layer=5)

    assert not torch.allclose(layer0, layer5)


def test_all_layers_accessible(model):
    """Test that all layers can be extracted."""

    prompt = "Test"
    n_layers = model.cfg.n_layers

    for layer in range(n_layers):
        attn = extract_attention_output(model, prompt, layer)
        assert attn.shape[2] == model.cfg.n_heads
        assert attn.shape[3] == model.cfg.d_head


def test_prompt_length_affects_sequence(model):
    """Test that prompt length affects sequence dimension."""

    short = extract_attention_output(model, "Hi", layer=0)
    long = extract_attention_output(model, "The quick brown fox", layer=0)

    assert short.shape[1] < long.shape[1]


def test_different_prompts_different_outputs(model):
    """Test that different prompts produce different outputs."""

    attn1 = extract_attention_output(model, "Hello", layer=0)
    attn2 = extract_attention_output(model, "Goodbye", layer=0)

    assert not torch.allclose(attn1, attn2)


def test_attention_values_finite(model):
    """Test that attention outputs have finite values."""

    attn = extract_attention_output(model, "Test", layer=0)
    assert torch.isfinite(attn).all()


def test_multi_head_structure(model):
    """Test that different heads have different values."""

    attn = extract_attention_output(model, "Test input", layer=0)
    # Extract two different heads
    head0 = attn[:, :, 0, :]
    head1 = attn[:, :, 1, :]

    # Different heads should have different values
    assert not torch.allclose(head0, head1)
