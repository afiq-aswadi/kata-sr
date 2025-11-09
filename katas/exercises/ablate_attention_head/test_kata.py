"""Tests for ablate_attention_head kata."""

import pytest
import torch
from transformer_lens import HookedTransformer

try:
    from user_kata import ablate_attention_head
except ImportError:
    from .reference import ablate_attention_head


@pytest.fixture(scope="module")
def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


def test_output_shape(model):
    """Output should have correct logits shape."""
    text = "Hello world"
    layer = 0
    head = 0

    logits = ablate_attention_head(model, text, layer, head)

    tokens = model.to_tokens(text)
    assert logits.shape[0] == 1  # batch
    assert logits.shape[1] == tokens.shape[1]  # seq length
    assert logits.shape[2] == model.cfg.d_vocab  # vocabulary size


def test_ablation_changes_output(model):
    """Ablating a head should change model output."""
    text = "The cat sat on the"
    layer = 5
    head = 9  # Known important head in GPT-2

    # Get normal output
    tokens = model.to_tokens(text)
    normal_logits = model(tokens)

    # Get ablated output
    ablated_logits = ablate_attention_head(model, text, layer, head)

    # Outputs should differ
    assert not torch.allclose(normal_logits, ablated_logits, atol=1e-4)


def test_different_heads_different_results(model):
    """Ablating different heads should give different results."""
    text = "Hello"
    layer = 3

    logits_head_0 = ablate_attention_head(model, text, layer, 0)
    logits_head_5 = ablate_attention_head(model, text, layer, 5)

    assert not torch.allclose(logits_head_0, logits_head_5, atol=1e-5)


def test_different_layers_different_results(model):
    """Ablating same head in different layers gives different results."""
    text = "Test input"
    head = 0

    logits_layer_0 = ablate_attention_head(model, text, 0, head)
    logits_layer_5 = ablate_attention_head(model, text, 5, head)

    assert not torch.allclose(logits_layer_0, logits_layer_5, atol=1e-5)


def test_first_head_first_layer(model):
    """Can ablate first head in first layer."""
    text = "Hi there"
    logits = ablate_attention_head(model, text, layer=0, head=0)

    assert logits.shape[2] == model.cfg.d_vocab


def test_last_head_last_layer(model):
    """Can ablate last head in last layer."""
    text = "Test"
    last_layer = model.cfg.n_layers - 1
    last_head = model.cfg.n_heads - 1

    logits = ablate_attention_head(model, text, last_layer, last_head)

    assert logits.shape[2] == model.cfg.d_vocab


def test_short_sequence(model):
    """Works with short sequences."""
    text = "A"
    logits = ablate_attention_head(model, text, layer=2, head=3)

    assert logits.shape[1] > 0


def test_longer_sequence(model):
    """Works with longer sequences."""
    text = "This is a longer sequence with many tokens"
    logits = ablate_attention_head(model, text, layer=4, head=7)

    tokens = model.to_tokens(text)
    assert logits.shape[1] == tokens.shape[1]
