"""Tests for hook_capture_activation kata."""

import torch
from transformer_lens import HookedTransformer

from framework import assert_shape

try:
    from user_kata import hook_capture_activation
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    hook_capture_activation = reference.hook_capture_activation  # type: ignore


def test_capture_residual_stream():
    """Should capture residual stream activations."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    text = "Hello world"

    activation = hook_capture_activation(model, text, "blocks.0.hook_resid_post")

    # Shape should be (batch=1, seq_len, d_model)
    tokens = model.to_tokens(text)
    seq_len = tokens.shape[1]
    assert_shape(activation, (1, seq_len, model.cfg.d_model), "residual stream")


def test_capture_attention_patterns():
    """Should capture attention patterns."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    text = "The cat sat"

    activation = hook_capture_activation(model, text, "blocks.0.attn.hook_pattern")

    # Shape should be (batch=1, n_heads, seq_len, seq_len)
    tokens = model.to_tokens(text)
    seq_len = tokens.shape[1]
    assert_shape(
        activation,
        (1, model.cfg.n_heads, seq_len, seq_len),
        "attention patterns"
    )


def test_capture_mlp_output():
    """Should capture MLP output."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    text = "Test"

    activation = hook_capture_activation(model, text, "blocks.0.hook_mlp_out")

    tokens = model.to_tokens(text)
    seq_len = tokens.shape[1]
    assert_shape(activation, (1, seq_len, model.cfg.d_model), "MLP output")


def test_capture_different_layers():
    """Should capture activations from different layers."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    text = "Hello"

    for layer in [0, 5, 11]:
        activation = hook_capture_activation(
            model, text, f"blocks.{layer}.hook_resid_post"
        )
        tokens = model.to_tokens(text)
        seq_len = tokens.shape[1]
        assert_shape(
            activation,
            (1, seq_len, model.cfg.d_model),
            f"layer {layer}"
        )


def test_no_nans_or_infs():
    """Captured activations should not contain NaNs or infinities."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    text = "Hello world"

    activation = hook_capture_activation(model, text, "blocks.0.hook_resid_post")

    assert not torch.isnan(activation).any(), "activation should not contain NaN"
    assert not torch.isinf(activation).any(), "activation should not contain inf"
