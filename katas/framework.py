"""Optional utilities for kata authors.

Keep this minimal - katas should be standalone Python code.
"""

import torch
from jaxtyping import Float


def assert_shape(
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
    name: str = "tensor",
) -> None:
    """Helper for shape assertions in tests.

    Args:
        tensor: tensor to check
        expected_shape: expected shape tuple
        name: tensor name for error message

    Raises:
        AssertionError: if shapes don't match
    """
    if tensor.shape != expected_shape:
        raise AssertionError(
            f"{name} has wrong shape.\n  Expected: {expected_shape}\n  Got: {tensor.shape}"
        )


def assert_close(
    a: Float[torch.Tensor, "..."],
    b: Float[torch.Tensor, "..."],
    rtol: float = 1e-5,
    atol: float = 1e-8,
    name: str = "tensor",
) -> None:
    """Helper for numerical comparison.

    Args:
        a: first tensor
        b: second tensor
        rtol: relative tolerance
        atol: absolute tolerance
        name: tensor name for error message

    Raises:
        AssertionError: if values don't match within tolerance
    """
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        max_diff = (a - b).abs().max().item()
        raise AssertionError(
            f"{name} values don't match.\n"
            f"  Max difference: {max_diff}\n"
            f"  Relative tolerance: {rtol}\n"
            f"  Absolute tolerance: {atol}"
        )
