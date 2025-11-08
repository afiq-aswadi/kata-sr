"""Tests for tensor slicing kata."""

import torch


def test_gather_rows():
    from template import gather_rows

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    indices = torch.tensor([2, 0, 1])
    result = gather_rows(x, indices)
    expected = torch.tensor([[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]])
    assert torch.allclose(result, expected)


def test_scatter_add():
    from template import scatter_add

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    indices = torch.tensor([0, 2, 0])
    result = scatter_add(x, indices, output_size=3)
    expected = torch.tensor([[6.0, 8.0], [0.0, 0.0], [3.0, 4.0]])
    assert torch.allclose(result, expected)


def test_scatter_add_empty_bins():
    from template import scatter_add

    x = torch.tensor([[1.0], [2.0]])
    indices = torch.tensor([0, 0])
    result = scatter_add(x, indices, output_size=5)
    assert result.shape == (5, 1)
    assert result[0, 0] == 3.0
    assert result[1:].sum() == 0.0


def test_masked_select_2d():
    from template import masked_select_2d

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[True, False], [False, True]])
    result = masked_select_2d(x, mask)
    expected = torch.tensor([1.0, 4.0])
    assert torch.allclose(result, expected)


def test_top_k_indices():
    from template import top_k_indices

    x = torch.tensor([[3.0, 1.0, 4.0, 2.0], [8.0, 5.0, 7.0, 6.0]])
    result = top_k_indices(x, k=2)
    expected = torch.tensor([[2, 0], [0, 2]])  # indices of top-2 per row
    assert torch.equal(result, expected)


def test_top_k_single_row():
    from template import top_k_indices

    x = torch.tensor([[10.0, 5.0, 15.0, 3.0, 20.0]])
    result = top_k_indices(x, k=3)
    expected = torch.tensor([[4, 2, 0]])  # 20, 15, 10
    assert torch.equal(result, expected)
