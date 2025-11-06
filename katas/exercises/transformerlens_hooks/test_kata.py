"""Tests for TransformerLens hooks kata."""

import pytest
import torch
from transformer_lens import HookedTransformer


@pytest.fixture
def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


def test_extract_attention_patterns(model):
    from template import extract_attention_patterns

    text = "The cat sat on the mat"
    patterns = extract_attention_patterns(model, text, layer=0)

    # Should have shape (batch, heads, seq, seq)
    assert patterns.dim() == 4
    assert patterns.shape[1] == model.cfg.n_heads

    # Attention should sum to 1 over last dimension
    assert torch.allclose(patterns.sum(dim=-1), torch.ones_like(patterns.sum(dim=-1)), atol=1e-5)


def test_get_residual_stream(model):
    from template import get_residual_stream

    text = "Hello world"
    residual = get_residual_stream(model, text, layer=0, position=0)

    # Should be a 1D vector of size d_model
    assert residual.dim() == 1
    assert residual.shape[0] == model.cfg.d_model


def test_register_activation_hook(model):
    from template import register_activation_hook

    hook_point = "blocks.0.attn.hook_q"
    activations = register_activation_hook(model, hook_point)

    # Run model
    model("Test input")

    # Should have captured activations
    assert len(activations) > 0
    assert isinstance(activations[0], torch.Tensor)

    # Clean up
    model.reset_hooks()


def test_mean_ablate_head(model):
    from template import mean_ablate_head

    text = "The quick brown fox"

    # Get normal output
    normal_logits = model(text)

    # Get ablated output
    ablated_logits = mean_ablate_head(model, text, layer=0, head=0)

    # Outputs should be different
    assert not torch.allclose(normal_logits, ablated_logits)

    # Shape should be same
    assert normal_logits.shape == ablated_logits.shape

    # Clean up
    model.reset_hooks()


def test_extract_multiple_layers(model):
    from template import extract_attention_patterns

    text = "Test"
    patterns_0 = extract_attention_patterns(model, text, layer=0)
    patterns_1 = extract_attention_patterns(model, text, layer=1)

    # Different layers should have different patterns
    assert not torch.allclose(patterns_0, patterns_1)
