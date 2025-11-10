"""Tests for merge sorted arrays kata."""

import pytest


def test_merge_sorted_basic():
    from template import merge_sorted

    result = merge_sorted([1, 3, 5], [2, 4, 6])
    assert result == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_empty_first():
    from template import merge_sorted

    result = merge_sorted([], [1, 2, 3])
    assert result == [1, 2, 3]


def test_merge_sorted_empty_second():
    from template import merge_sorted

    result = merge_sorted([1, 2, 3], [])
    assert result == [1, 2, 3]


def test_merge_sorted_different_lengths():
    from template import merge_sorted

    result = merge_sorted([1, 2], [3, 4, 5, 6])
    assert result == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_overlapping():
    from template import merge_sorted

    result = merge_sorted([1, 4, 7], [2, 3, 5])
    assert result == [1, 2, 3, 4, 5, 7]


def test_merge_in_place_basic():
    from template import merge_in_place

    nums1 = [1, 2, 3, 0, 0, 0]
    merge_in_place(nums1, 3, [2, 5, 6], 3)
    assert nums1 == [1, 2, 2, 3, 5, 6]


def test_merge_in_place_empty_first():
    from template import merge_in_place

    nums1 = [0, 0, 0]
    merge_in_place(nums1, 0, [1, 2, 3], 3)
    assert nums1 == [1, 2, 3]


def test_merge_in_place_empty_second():
    from template import merge_in_place

    nums1 = [1, 2, 3]
    merge_in_place(nums1, 3, [], 0)
    assert nums1 == [1, 2, 3]


def test_merge_in_place_interleaved():
    from template import merge_in_place

    nums1 = [1, 3, 5, 0, 0, 0]
    merge_in_place(nums1, 3, [2, 4, 6], 3)
    assert nums1 == [1, 2, 3, 4, 5, 6]
