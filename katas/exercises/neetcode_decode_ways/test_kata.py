"""Tests for Decode Ways kata."""

def test_num_decodings_example1():
    from template import num_decodings
    assert num_decodings("12") == 2

def test_num_decodings_example2():
    from template import num_decodings
    assert num_decodings("226") == 3

def test_num_decodings_example3():
    from template import num_decodings
    assert num_decodings("06") == 0

def test_num_decodings_single():
    from template import num_decodings
    assert num_decodings("1") == 1
    assert num_decodings("0") == 0

def test_num_decodings_multiple_zeros():
    from template import num_decodings
    assert num_decodings("10") == 1
    assert num_decodings("27") == 1
