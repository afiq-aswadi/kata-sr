"""Conv2D kata - reference solution."""

import torch
import torch.nn.functional as F
from jaxtyping import Float


def conv2d(
    x: Float[torch.Tensor, "batch in_c h w"],
    kernel: Float[torch.Tensor, "out_c in_c kh kw"],
    stride: int = 1,
    padding: int = 0,
) -> Float[torch.Tensor, "batch out_c h_out w_out"]:
    """Apply 2D convolution to input tensor."""
    return F.conv2d(x, kernel, stride=stride, padding=padding)


def conv2d_einsum(
    x: Float[torch.Tensor, "batch in_c h w"],
    kernel: Float[torch.Tensor, "out_c in_c kh kw"],
) -> Float[torch.Tensor, "batch out_c h_out w_out"]:
    """Simplified conv2d using einsum (no stride/padding)."""
    batch, in_c, h, w = x.shape
    out_c, _, kh, kw = kernel.shape

    # Pad input
    x_padded = x

    # Use unfold to create sliding windows
    x_unfolded = F.unfold(x_padded, kernel_size=(kh, kw))
    # x_unfolded: (batch, in_c * kh * kw, num_windows)

    # Reshape for einsum
    num_windows = x_unfolded.shape[-1]
    h_out = h - kh + 1
    w_out = w - kw + 1

    x_unfolded = x_unfolded.view(batch, in_c, kh * kw, h_out * w_out)
    kernel_flat = kernel.view(out_c, in_c, kh * kw)

    # Einsum: batch, in_channels, kernel_size, positions
    result = torch.einsum("biks,ois->boks", x_unfolded, kernel_flat)
    return result.view(batch, out_c, h_out, w_out)
