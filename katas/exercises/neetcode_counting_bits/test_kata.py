"""Tests for Counting Bits kata."""

try:
    from user_kata import count_bits
except ImportError:
    from .reference import count_bits


def test_count_bits_example1():
    assert count_bits(2) == [0,1,1]

def test_count_bits_example2():
    assert count_bits(5) == [0,1,1,2,1,2]

def test_count_bits_zero():
    assert count_bits(0) == [0]

def test_count_bits_one():
    assert count_bits(1) == [0,1]

def test_count_bits_larger():
    assert count_bits(8) == [0,1,1,2,1,2,2,3,1]
