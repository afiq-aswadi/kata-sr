"""Tests for Decode Ways kata."""

try:
    from user_kata import num_decodings
except ImportError:
    from .reference import num_decodings


def test_num_decodings_example1():
    assert num_decodings("12") == 2

def test_num_decodings_example2():
    assert num_decodings("226") == 3

def test_num_decodings_example3():
    assert num_decodings("06") == 0

def test_num_decodings_single():
    assert num_decodings("1") == 1
    assert num_decodings("0") == 0

def test_num_decodings_multiple_zeros():
    assert num_decodings("10") == 1
    assert num_decodings("27") == 1
