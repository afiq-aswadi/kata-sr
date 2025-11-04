"""Tests for multi-head attention kata."""


import torch
import torch.nn.functional as F

from framework import assert_close, assert_shape

try:
    from user_kata import MultiHeadAttention, create_causal_mask
except ModuleNotFoundError:  # pragma: no cover - fallback for standalone test runs
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("multihead_attention_reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)

    MultiHeadAttention = reference.MultiHeadAttention  # type: ignore[attr-defined]
    create_causal_mask = reference.create_causal_mask  # type: ignore[attr-defined]


def _linear(x: torch.Tensor, layer: torch.nn.Linear) -> torch.Tensor:
    """Apply a linear layer to the input without triggering autograd surprises."""

    return F.linear(x, layer.weight, layer.bias)


def _manual_attention(
    module: MultiHeadAttention,
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute attention output using the module's parameters."""

    batch, seq_len, _ = x.shape

    q = _linear(x, module.w_q)
    k = _linear(x, module.w_k)
    v = _linear(x, module.w_v)

    q = q.view(batch, seq_len, module.num_heads, module.d_head).transpose(1, 2)
    k = k.view(batch, seq_len, module.num_heads, module.d_head).transpose(1, 2)
    v = v.view(batch, seq_len, module.num_heads, module.d_head).transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / (module.d_head**0.5)

    if mask is not None:
        mask_bool = mask.to(dtype=torch.bool)
        if mask_bool.dim() == 2:
            mask_expanded = mask_bool.unsqueeze(0).unsqueeze(0)
        elif mask_bool.dim() == 3:
            mask_expanded = mask_bool.unsqueeze(1)
        else:
            raise ValueError("mask must be 2D or 3D tensor")
        scores = scores.masked_fill(mask_expanded, torch.finfo(scores.dtype).min)

    attn_weights = torch.softmax(scores, dim=-1)
    attended = torch.matmul(attn_weights, v)

    attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, module.d_model)
    return _linear(attended, module.w_o)


def _set_deterministic_weights(module: MultiHeadAttention, scale: float = 0.01) -> None:
    """Fill projection matrices with deterministic values and zero biases."""

    with torch.no_grad():
        base = torch.arange(module.d_model * module.d_model, dtype=torch.float32)
        base = (base * scale).view(module.d_model, module.d_model)

        for idx, layer in enumerate([module.w_q, module.w_k, module.w_v, module.w_o]):
            layer.weight.copy_(base + idx)
            if layer.bias is not None:
                layer.bias.zero_()


def test_output_shape():
    """Output should have same shape as input."""
    batch, seq_len, d_model = 2, 10, 64
    num_heads = 8

    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch, seq_len, d_model)
    out = mha(x)

    assert_shape(out, (batch, seq_len, d_model), "attention output")


def test_different_sequence_lengths():
    """Attention should handle different sequence lengths."""
    d_model, num_heads = 64, 8
    mha = MultiHeadAttention(d_model, num_heads)

    for seq_len in [1, 5, 10, 20]:
        x = torch.randn(2, seq_len, d_model)
        out = mha(x)
        assert_shape(out, (2, seq_len, d_model), f"seq_len={seq_len}")


def test_different_batch_sizes():
    """Attention should handle different batch sizes."""
    seq_len, d_model, num_heads = 10, 64, 8
    mha = MultiHeadAttention(d_model, num_heads)

    for batch in [1, 4, 16]:
        x = torch.randn(batch, seq_len, d_model)
        out = mha(x)
        assert_shape(out, (batch, seq_len, d_model), f"batch={batch}")


def test_gradient_flow():
    """Gradients should flow through attention."""
    batch, seq_len, d_model = 2, 5, 16
    num_heads = 4

    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)

    out = mha(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "gradients should flow to input"
    assert not torch.isnan(x.grad).any(), "gradients should not be NaN"


def test_deterministic():
    """Same input should produce same output."""
    batch, seq_len, d_model = 2, 5, 16
    num_heads = 4

    torch.manual_seed(42)
    mha = MultiHeadAttention(d_model, num_heads)

    x = torch.randn(batch, seq_len, d_model)
    out1 = mha(x)
    out2 = mha(x)

    assert_close(out1, out2, name="deterministic output")


def test_different_num_heads():
    """Attention should work with different number of heads."""
    batch, seq_len, d_model = 2, 5, 64

    for num_heads in [1, 2, 4, 8]:
        mha = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch, seq_len, d_model)
        out = mha(x)
        assert_shape(out, (batch, seq_len, d_model), f"num_heads={num_heads}")


def test_single_head():
    """Attention should work with single head (degenerates to regular attention)."""
    batch, seq_len, d_model = 2, 5, 16
    mha = MultiHeadAttention(d_model, num_heads=1)

    x = torch.randn(batch, seq_len, d_model)
    out = mha(x)
    assert_shape(out, (batch, seq_len, d_model), "single head")


def test_no_nans_or_infs():
    """Output should not contain NaNs or infinities."""
    batch, seq_len, d_model = 2, 10, 64
    num_heads = 8

    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch, seq_len, d_model)
    out = mha(x)

    assert not torch.isnan(out).any(), "output should not contain NaN"
    assert not torch.isinf(out).any(), "output should not contain inf"


def test_attention_matches_manual_computation():
    """Attention output should match manual computation with set weights."""
    d_model, num_heads = 8, 2
    batch, seq_len = 2, 3

    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model, num_heads)
    _set_deterministic_weights(mha)

    x = torch.randn(batch, seq_len, d_model)
    expected = _manual_attention(mha, x)
    out = mha(x)

    assert_close(out, expected, atol=1e-6, rtol=1e-5, name="attention output")


def test_causal_mask_applied():
    """Applying a causal mask should prevent attending to future positions."""
    d_model, num_heads = 6, 3
    seq_len = 4

    mha = MultiHeadAttention(d_model, num_heads)
    _set_deterministic_weights(mha, scale=0.02)

    x = torch.arange(seq_len * d_model, dtype=torch.float32).view(1, seq_len, d_model)
    mask = create_causal_mask(seq_len)

    out_masked = mha(x, mask=mask)
    manual_masked = _manual_attention(mha, x, mask=mask)

    assert_close(out_masked, manual_masked, atol=1e-6, rtol=1e-5, name="masked attention")

    out_unmasked = mha(x)
    assert not torch.allclose(out_masked, out_unmasked), "causal mask should change attention"


def test_causal_mask_shape():
    """Causal mask should have correct shape."""
    seq_len = 5
    mask = create_causal_mask(seq_len)
    assert_shape(mask, (seq_len, seq_len), "causal mask")


def test_causal_mask_structure():
    """Causal mask should be upper triangular."""
    mask = create_causal_mask(4)
    expected = torch.tensor(
        [
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ]
    )
    assert torch.equal(mask, expected), "causal mask should be upper triangular"


def test_causal_mask_single_element():
    """Causal mask for single element should be all False."""
    mask = create_causal_mask(1)
    expected = torch.tensor([[False]])
    assert torch.equal(mask, expected), "single element mask"
