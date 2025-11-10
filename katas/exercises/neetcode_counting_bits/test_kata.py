"""Tests for Counting Bits kata."""

def test_count_bits_example1():
    from template import count_bits
    assert count_bits(2) == [0,1,1]

def test_count_bits_example2():
    from template import count_bits
    assert count_bits(5) == [0,1,1,2,1,2]

def test_count_bits_zero():
    from template import count_bits
    assert count_bits(0) == [0]

def test_count_bits_one():
    from template import count_bits
    assert count_bits(1) == [0,1]

def test_count_bits_larger():
    from template import count_bits
    assert count_bits(8) == [0,1,1,2,1,2,2,3,1]
