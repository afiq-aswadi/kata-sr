"""Tests for Minimum Window Substring kata."""

def test_min_window_example1():
    from template import min_window
    assert min_window("ADOBECODEBANC", "ABC") == "BANC"

def test_min_window_example2():
    from template import min_window
    assert min_window("a", "a") == "a"

def test_min_window_example3():
    from template import min_window
    assert min_window("a", "aa") == ""
