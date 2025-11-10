"""Tests for Sliding Window Maximum kata."""

def test_max_sliding_window_example1():
    from template import max_sliding_window
    assert max_sliding_window([1,3,-1,-3,5,3,6,7], 3) == [3,3,5,5,6,7]

def test_max_sliding_window_example2():
    from template import max_sliding_window
    assert max_sliding_window([1], 1) == [1]

def test_max_sliding_window_decreasing():
    from template import max_sliding_window
    assert max_sliding_window([7,6,5,4,3,2,1], 3) == [7,6,5,4,3]
