"""Tests for extract attention patterns kata."""

import pytest
import torch
from transformer_lens import HookedTransformer

try:
    from user_kata import extract_attention_patterns
except ImportError:
    from .reference import extract_attention_patterns


@pytest.fixture
def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


def test_output_shape(model):
    """Verify output has shape (batch, n_heads, seq, seq)"""
    text = "The cat sat on the mat"
    patterns = extract_attention_patterns(model, text, layer=0)

    assert patterns.dim() == 4
    assert patterns.shape[0] == 1  # batch
    assert patterns.shape[1] == model.cfg.n_heads
    assert patterns.shape[2] == patterns.shape[3]  # seq_q == seq_k


def test_patterns_normalized(model):
    """Attention patterns should sum to 1.0 across key dimension"""
    text = "Hello world"
    patterns = extract_attention_patterns(model, text, layer=0)

    sums = patterns.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_patterns_range(model):
    """Attention weights should be in [0, 1] (probabilities)"""
    text = "Testing range"
    patterns = extract_attention_patterns(model, text, layer=0)

    assert (patterns >= 0).all()
    assert (patterns <= 1).all()


def test_different_layers(model):
    """Different layers should produce different patterns"""
    text = "Testing different layers"
    patterns_0 = extract_attention_patterns(model, text, layer=0)
    patterns_5 = extract_attention_patterns(model, text, layer=5)

    assert not torch.allclose(patterns_0, patterns_5, atol=1e-3)


def test_single_token(model):
    """Handle single token input"""
    text = "Hi"
    patterns = extract_attention_patterns(model, text, layer=0)

    # Single token attends to itself with weight 1.0
    assert patterns.shape[2] == 1
    assert patterns.shape[3] == 1


def test_sequence_length_matches(model):
    """Sequence dimension should match tokenized length"""
    text = "One two three four"
    patterns = extract_attention_patterns(model, text, layer=0)

    # Tokenize to get expected length
    tokens = model.to_tokens(text)
    expected_seq_len = tokens.shape[1]

    assert patterns.shape[2] == expected_seq_len
    assert patterns.shape[3] == expected_seq_len


def test_layer_consistency(model):
    """Same layer, same input should give same output"""
    text = "Consistency test"
    patterns_1 = extract_attention_patterns(model, text, layer=3)
    patterns_2 = extract_attention_patterns(model, text, layer=3)

    assert torch.allclose(patterns_1, patterns_2, atol=1e-6)
