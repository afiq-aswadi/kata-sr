"""Conv2D implementation kata."""

import torch
from jaxtyping import Float


def conv2d(
    x: Float[torch.Tensor, "batch in_c h w"],
    kernel: Float[torch.Tensor, "out_c in_c kh kw"],
    stride: int = 1,
    padding: int = 0,
) -> Float[torch.Tensor, "batch out_c h_out w_out"]:
    """Apply 2D convolution to input tensor.

    Args:
        x: input tensor (batch, in_channels, height, width)
        kernel: convolution kernel (out_channels, in_channels, kernel_h, kernel_w)
        stride: stride for convolution
        padding: zero-padding amount

    Returns:
        convolved tensor
    """
    # TODO: implement conv2d using unfold or manual sliding window
    # Hint: torch.nn.functional.unfold can help, or use einsum
    # BLANK_START
    pass
    # BLANK_END


def conv2d_einsum(
    x: Float[torch.Tensor, "batch in_c h w"],
    kernel: Float[torch.Tensor, "out_c in_c kh kw"],
) -> Float[torch.Tensor, "batch out_c h_out w_out"]:
    """Simplified conv2d using einsum (no stride/padding).

    Args:
        x: input tensor
        kernel: convolution kernel

    Returns:
        convolved tensor
    """
    # TODO: use unfold + einsum for clean implementation
    # BLANK_START
    pass
    # BLANK_END
