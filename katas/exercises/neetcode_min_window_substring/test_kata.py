"""Tests for Minimum Window Substring kata."""

try:
    from user_kata import min_window
except ImportError:
    from .reference import min_window


def test_min_window_example1():
    assert min_window("ADOBECODEBANC", "ABC") == "BANC"

def test_min_window_example2():
    assert min_window("a", "a") == "a"

def test_min_window_example3():
    assert min_window("a", "aa") == ""
