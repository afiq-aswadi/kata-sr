"""Tests for MLP kata."""

import torch

from framework import assert_shape

try:
    from user_kata import MLP
except ModuleNotFoundError:  # pragma: no cover - fallback for standalone test runs
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("mlp_reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)

    MLP = reference.MLP  # type: ignore[attr-defined]


def test_output_shape():
    """Output should have correct shape."""
    mlp = MLP(input_dim=10, hidden_dims=[20, 15], output_dim=5)
    x = torch.randn(3, 10)
    out = mlp(x)
    assert_shape(out, (3, 5), "mlp output")


def test_single_hidden_layer():
    """MLP should work with single hidden layer."""
    mlp = MLP(input_dim=5, hidden_dims=[10], output_dim=3)
    x = torch.randn(2, 5)
    out = mlp(x)
    assert_shape(out, (2, 3), "single hidden layer output")


def test_no_hidden_layers():
    """MLP should work with no hidden layers (just input->output)."""
    mlp = MLP(input_dim=5, hidden_dims=[], output_dim=3)
    x = torch.randn(2, 5)
    out = mlp(x)
    assert_shape(out, (2, 3), "no hidden layers output")


def test_multiple_hidden_layers():
    """MLP should work with multiple hidden layers."""
    mlp = MLP(input_dim=10, hidden_dims=[20, 15, 10, 5], output_dim=2)
    x = torch.randn(4, 10)
    out = mlp(x)
    assert_shape(out, (4, 2), "multiple hidden layers output")


def test_gradient_flow():
    """Gradients should flow through all layers."""
    mlp = MLP(input_dim=5, hidden_dims=[10], output_dim=3)
    x = torch.randn(2, 5, requires_grad=True)
    out = mlp(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "gradients should flow to input"
    assert not torch.isnan(x.grad).any(), "gradients should not be NaN"


def test_no_activation_on_output():
    """Output layer should not have ReLU (allows negative values)."""
    mlp = MLP(input_dim=5, hidden_dims=[10], output_dim=3)

    # with enough random runs, we should see negative outputs
    has_negative = False
    torch.manual_seed(42)
    for _ in range(100):
        x = torch.randn(1, 5)
        out = mlp(x)
        if (out < 0).any():
            has_negative = True
            break

    assert has_negative, "output should allow negative values (no ReLU on output layer)"


def test_deterministic():
    """Same input should produce same output."""
    torch.manual_seed(42)
    mlp = MLP(input_dim=5, hidden_dims=[10], output_dim=3)

    x = torch.randn(2, 5)
    out1 = mlp(x)
    out2 = mlp(x)

    assert torch.allclose(out1, out2), "output should be deterministic"


def test_different_batch_sizes():
    """MLP should handle different batch sizes."""
    mlp = MLP(input_dim=5, hidden_dims=[10], output_dim=3)

    x1 = torch.randn(1, 5)
    out1 = mlp(x1)
    assert_shape(out1, (1, 3), "batch size 1")

    x2 = torch.randn(10, 5)
    out2 = mlp(x2)
    assert_shape(out2, (10, 3), "batch size 10")

    x3 = torch.randn(100, 5)
    out3 = mlp(x3)
    assert_shape(out3, (100, 3), "batch size 100")


def test_parameters_trainable():
    """All parameters should be trainable."""
    mlp = MLP(input_dim=5, hidden_dims=[10, 8], output_dim=3)

    params = list(mlp.parameters())
    assert len(params) > 0, "MLP should have parameters"

    for param in params:
        assert param.requires_grad, "all parameters should be trainable"


def test_forward_pass_non_zero():
    """Forward pass should produce non-zero outputs."""
    torch.manual_seed(42)
    mlp = MLP(input_dim=5, hidden_dims=[10], output_dim=3)
    x = torch.randn(2, 5)
    out = mlp(x)

    # with random initialization and random input, output shouldn't be all zeros
    assert not torch.allclose(out, torch.zeros_like(out)), "output should be non-zero"
