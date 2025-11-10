"""Tests for maximum subarray kata."""

import pytest


def test_max_subarray_basic():
    from template import max_subarray_sum

    result = max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4])
    assert result == 6  # [4, -1, 2, 1]


def test_max_subarray_all_negative():
    from template import max_subarray_sum

    result = max_subarray_sum([-2, -3, -1, -5])
    assert result == -1


def test_max_subarray_all_positive():
    from template import max_subarray_sum

    result = max_subarray_sum([1, 2, 3, 4])
    assert result == 10


def test_max_subarray_single_element():
    from template import max_subarray_sum

    result = max_subarray_sum([5])
    assert result == 5


def test_max_subarray_mixed():
    from template import max_subarray_sum

    result = max_subarray_sum([5, -3, 5])
    assert result == 7


def test_max_subarray_indices_basic():
    from template import max_subarray_indices

    result = max_subarray_indices([-2, 1, -3, 4, -1, 2, 1, -5, 4])
    assert result == (3, 6)  # indices of [4, -1, 2, 1]


def test_max_subarray_indices_single():
    from template import max_subarray_indices

    result = max_subarray_indices([5])
    assert result == (0, 0)


def test_max_subarray_indices_full_array():
    from template import max_subarray_indices

    result = max_subarray_indices([1, 2, 3, 4])
    assert result == (0, 3)
