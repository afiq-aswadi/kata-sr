"""Tests for Median of Two Sorted Arrays kata."""

try:
    from user_kata import find_median_sorted_arrays
except ImportError:
    from .reference import find_median_sorted_arrays


def test_median_example1():
    assert find_median_sorted_arrays([1,3], [2]) == 2.0

def test_median_example2():
    assert find_median_sorted_arrays([1,2], [3,4]) == 2.5

def test_median_empty_array():
    assert find_median_sorted_arrays([], [1]) == 1.0

def test_median_single_elements():
    assert find_median_sorted_arrays([2], [1,3,4]) == 2.5

def test_median_different_sizes():
    result = find_median_sorted_arrays([1,2,3,4,5], [6,7,8,9,10])
    assert result == 5.5

def test_median_negative_numbers():
    assert find_median_sorted_arrays([-5,-3,-1], [0,2,4]) == -0.5
