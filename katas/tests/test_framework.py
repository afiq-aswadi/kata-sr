"""Tests for framework helper functions."""

import pytest
import torch

from framework import assert_close, assert_shape


def test_assert_shape_correct():
    """Test assert_shape with correct shape."""
    x = torch.randn(3, 4, 5)
    # should not raise
    assert_shape(x, (3, 4, 5))


def test_assert_shape_wrong():
    """Test assert_shape with incorrect shape."""
    x = torch.randn(3, 4, 5)
    with pytest.raises(AssertionError) as exc_info:
        assert_shape(x, (3, 4, 6))

    error_msg = str(exc_info.value)
    assert "Expected: (3, 4, 6)" in error_msg
    assert "Got: torch.Size([3, 4, 5])" in error_msg


def test_assert_shape_custom_name():
    """Test assert_shape with custom tensor name."""
    x = torch.randn(2, 3)
    with pytest.raises(AssertionError) as exc_info:
        assert_shape(x, (2, 4), name="attention_weights")

    error_msg = str(exc_info.value)
    assert "attention_weights has wrong shape" in error_msg


def test_assert_close_matching():
    """Test assert_close with matching tensors."""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 3.0])
    # should not raise
    assert_close(a, b)


def test_assert_close_within_tolerance():
    """Test assert_close with values within tolerance."""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.000001, 2.000001, 3.000001])
    # should not raise with default rtol=1e-5
    assert_close(a, b)


def test_assert_close_outside_tolerance():
    """Test assert_close with values outside tolerance."""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.1, 2.0, 3.0])

    with pytest.raises(AssertionError) as exc_info:
        assert_close(a, b, rtol=1e-5)

    error_msg = str(exc_info.value)
    assert "values don't match" in error_msg
    assert "Max difference:" in error_msg
    assert "Relative tolerance:" in error_msg


def test_assert_close_custom_tolerance():
    """Test assert_close with custom tolerance."""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.1, 2.0, 3.0])

    # should not raise with larger tolerance
    assert_close(a, b, rtol=0.2)


def test_assert_close_custom_name():
    """Test assert_close with custom tensor name."""
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([2.0, 3.0])

    with pytest.raises(AssertionError) as exc_info:
        assert_close(a, b, name="output")

    error_msg = str(exc_info.value)
    assert "output values don't match" in error_msg


def test_assert_close_multidimensional():
    """Test assert_close with multidimensional tensors."""
    a = torch.randn(3, 4, 5)
    b = a.clone()
    # should not raise
    assert_close(a, b)

    # add small perturbation
    b = a + 1e-6
    assert_close(a, b, atol=1e-5)

    # add large perturbation
    b = a + 1.0
    with pytest.raises(AssertionError):
        assert_close(a, b)
