"""Tests for QK similarity kata."""

import torch

from framework import assert_close, assert_shape

try:
    from user_kata import compute_qk_similarity
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    compute_qk_similarity = reference.compute_qk_similarity  # type: ignore


def test_output_shape():
    """Output should have shape (batch, seq_q, seq_k)."""
    batch, seq_q, seq_k, d_model = 2, 5, 7, 64
    Q = torch.randn(batch, seq_q, d_model)
    K = torch.randn(batch, seq_k, d_model)

    scores = compute_qk_similarity(Q, K)
    assert_shape(scores, (batch, seq_q, seq_k), "attention scores")


def test_self_attention_shape():
    """Self-attention should produce square score matrix."""
    batch, seq_len, d_model = 2, 10, 32
    Q = K = torch.randn(batch, seq_len, d_model)

    scores = compute_qk_similarity(Q, K)
    assert_shape(scores, (batch, seq_len, seq_len), "self-attention scores")


def test_scaling_applied():
    """Scores should be scaled by sqrt(d_model)."""
    batch, seq_len, d_model = 1, 3, 16
    Q = K = torch.ones(batch, seq_len, d_model)

    scores = compute_qk_similarity(Q, K)
    # Q @ K^T = d_model * ones, scaled by sqrt(d_model) = sqrt(d_model) * ones
    expected = torch.ones(batch, seq_len, seq_len) * (d_model**0.5)
    assert_close(scores, expected, name="scaled scores")


def test_different_dimensions():
    """Should work with different d_model dimensions."""
    batch, seq_len = 2, 5

    for d_model in [8, 32, 64, 128]:
        Q = K = torch.randn(batch, seq_len, d_model)
        scores = compute_qk_similarity(Q, K)
        assert_shape(scores, (batch, seq_len, seq_len), f"d_model={d_model}")


def test_cross_attention():
    """Should handle different sequence lengths (cross-attention)."""
    batch, seq_q, seq_k, d_model = 2, 3, 5, 32
    Q = torch.randn(batch, seq_q, d_model)
    K = torch.randn(batch, seq_k, d_model)

    scores = compute_qk_similarity(Q, K)
    assert_shape(scores, (batch, seq_q, seq_k), "cross-attention scores")


def test_deterministic():
    """Same inputs should produce same outputs."""
    batch, seq_len, d_model = 2, 4, 32

    torch.manual_seed(42)
    Q = K = torch.randn(batch, seq_len, d_model)

    scores1 = compute_qk_similarity(Q, K)
    scores2 = compute_qk_similarity(Q, K)

    assert_close(scores1, scores2, name="deterministic output")


def test_no_nans_or_infs():
    """Output should not contain NaNs or infinities."""
    batch, seq_len, d_model = 2, 5, 64
    Q = K = torch.randn(batch, seq_len, d_model)

    scores = compute_qk_similarity(Q, K)

    assert not torch.isnan(scores).any(), "output should not contain NaN"
    assert not torch.isinf(scores).any(), "output should not contain inf"
