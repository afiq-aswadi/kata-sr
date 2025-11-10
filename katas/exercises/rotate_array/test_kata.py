"""Tests for rotate array kata."""

import pytest


def test_rotate_basic():
    from template import rotate

    nums = [1, 2, 3, 4, 5, 6, 7]
    rotate(nums, 3)
    assert nums == [5, 6, 7, 1, 2, 3, 4]


def test_rotate_by_one():
    from template import rotate

    nums = [1, 2, 3]
    rotate(nums, 1)
    assert nums == [3, 1, 2]


def test_rotate_by_length():
    from template import rotate

    nums = [1, 2, 3, 4]
    rotate(nums, 4)
    assert nums == [1, 2, 3, 4]


def test_rotate_larger_than_length():
    from template import rotate

    nums = [1, 2, 3]
    rotate(nums, 5)  # 5 % 3 = 2
    assert nums == [2, 3, 1]


def test_rotate_empty():
    from template import rotate

    nums = []
    rotate(nums, 3)
    assert nums == []


def test_rotate_single_element():
    from template import rotate

    nums = [1]
    rotate(nums, 1)
    assert nums == [1]


def test_rotate_left_basic():
    from template import rotate_left

    nums = [1, 2, 3, 4, 5]
    rotate_left(nums, 2)
    assert nums == [3, 4, 5, 1, 2]


def test_rotate_left_by_one():
    from template import rotate_left

    nums = [1, 2, 3]
    rotate_left(nums, 1)
    assert nums == [2, 3, 1]
