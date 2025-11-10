"""Tests for Median of Two Sorted Arrays kata."""

def test_median_example1():
    from template import find_median_sorted_arrays
    assert find_median_sorted_arrays([1,3], [2]) == 2.0

def test_median_example2():
    from template import find_median_sorted_arrays
    assert find_median_sorted_arrays([1,2], [3,4]) == 2.5

def test_median_empty_array():
    from template import find_median_sorted_arrays
    assert find_median_sorted_arrays([], [1]) == 1.0

def test_median_single_elements():
    from template import find_median_sorted_arrays
    assert find_median_sorted_arrays([2], [1,3,4]) == 2.5

def test_median_different_sizes():
    from template import find_median_sorted_arrays
    result = find_median_sorted_arrays([1,2,3,4,5], [6,7,8,9,10])
    assert result == 5.5

def test_median_negative_numbers():
    from template import find_median_sorted_arrays
    assert find_median_sorted_arrays([-5,-3,-1], [0,2,4]) == -0.5
