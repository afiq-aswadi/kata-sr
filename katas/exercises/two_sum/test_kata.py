"""Tests for two sum kata."""

import pytest


def test_two_sum_basic():
    from template import two_sum

    result = two_sum([2, 7, 11, 15], 9)
    assert result == [0, 1]


def test_two_sum_negative_numbers():
    from template import two_sum

    result = two_sum([3, 2, -4], -2)
    assert result == [1, 2]


def test_two_sum_same_number():
    from template import two_sum

    result = two_sum([3, 3], 6)
    assert result == [0, 1]


def test_two_sum_larger_array():
    from template import two_sum

    result = two_sum([1, 5, 3, 7, 9, 2], 10)
    assert result in ([2, 4], [1, 3])  # 3+7 or 5+7


def test_two_sum_sorted_basic():
    from template import two_sum_sorted

    result = two_sum_sorted([1, 2, 3, 4, 5], 9)
    assert result == [3, 4]


def test_two_sum_sorted_negative():
    from template import two_sum_sorted

    result = two_sum_sorted([-3, -1, 0, 2, 4], 1)
    assert result == [2, 3]


def test_two_sum_sorted_first_and_last():
    from template import two_sum_sorted

    result = two_sum_sorted([1, 2, 3, 9], 10)
    assert result == [0, 3]
