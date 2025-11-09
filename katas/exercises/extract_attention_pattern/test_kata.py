"""Tests for extract_attention_pattern kata."""

import pytest
import torch
from transformer_lens import HookedTransformer

try:
    from user_kata import extract_attention_pattern
except ImportError:
    from .reference import extract_attention_pattern


@pytest.fixture(scope="module")
def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


def test_output_shape(model):
    """Verify output has correct shape (batch, n_heads, seq, seq)."""
    text = "Hello world"
    layer = 0
    patterns = extract_attention_pattern(model, text, layer)

    assert patterns.ndim == 4
    assert patterns.shape[0] == 1  # batch size
    assert patterns.shape[1] == model.cfg.n_heads
    assert patterns.shape[2] == patterns.shape[3]  # seq x seq


def test_patterns_normalized(model):
    """Attention patterns should sum to 1.0 across key dimension."""
    text = "The quick brown fox"
    layer = 1
    patterns = extract_attention_pattern(model, text, layer)

    sums = patterns.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_patterns_positive(model):
    """All attention weights should be non-negative."""
    text = "Test input"
    layer = 2
    patterns = extract_attention_pattern(model, text, layer)

    assert (patterns >= 0).all()


def test_patterns_bounded(model):
    """All attention weights should be <= 1.0."""
    text = "Another test"
    layer = 0
    patterns = extract_attention_pattern(model, text, layer)

    assert (patterns <= 1.0).all()


def test_different_layers(model):
    """Patterns from different layers should differ."""
    text = "Same input"
    patterns_0 = extract_attention_pattern(model, text, 0)
    patterns_5 = extract_attention_pattern(model, text, 5)

    assert not torch.allclose(patterns_0, patterns_5)


def test_short_sequence(model):
    """Handle short sequences (2-3 tokens)."""
    text = "Hi"
    layer = 0
    patterns = extract_attention_pattern(model, text, layer)

    assert patterns.shape[0] == 1
    assert patterns.shape[1] == model.cfg.n_heads
    assert patterns.shape[2] > 0  # At least 1 token


def test_long_sequence(model):
    """Handle longer sequences."""
    text = "This is a longer sequence with many more tokens in it"
    layer = 3
    patterns = extract_attention_pattern(model, text, layer)

    seq_len = patterns.shape[2]
    assert seq_len > 10  # Should have many tokens
    assert patterns.shape == (1, model.cfg.n_heads, seq_len, seq_len)


def test_last_layer(model):
    """Extract patterns from last layer."""
    text = "Test"
    last_layer = model.cfg.n_layers - 1
    patterns = extract_attention_pattern(model, text, last_layer)

    assert patterns.shape[1] == model.cfg.n_heads
