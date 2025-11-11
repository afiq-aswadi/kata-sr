"""Tests for maximum subarray kata."""

import pytest


try:
    from user_kata import max_subarray_sum
    from user_kata import max_subarray_indices
except ImportError:
    from .reference import max_subarray_sum
    from .reference import max_subarray_indices


def test_max_subarray_basic():

    result = max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4])
    assert result == 6  # [4, -1, 2, 1]


def test_max_subarray_all_negative():

    result = max_subarray_sum([-2, -3, -1, -5])
    assert result == -1


def test_max_subarray_all_positive():

    result = max_subarray_sum([1, 2, 3, 4])
    assert result == 10


def test_max_subarray_single_element():

    result = max_subarray_sum([5])
    assert result == 5


def test_max_subarray_mixed():

    result = max_subarray_sum([5, -3, 5])
    assert result == 7


def test_max_subarray_indices_basic():

    result = max_subarray_indices([-2, 1, -3, 4, -1, 2, 1, -5, 4])
    assert result == (3, 6)  # indices of [4, -1, 2, 1]


def test_max_subarray_indices_single():

    result = max_subarray_indices([5])
    assert result == (0, 0)


def test_max_subarray_indices_full_array():

    result = max_subarray_indices([1, 2, 3, 4])
    assert result == (0, 3)
