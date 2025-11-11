"""Tests for Reverse Bits kata."""

try:
    from user_kata import reverse_bits
except ImportError:
    from .reference import reverse_bits


def test_reverse_bits_example1():
    assert reverse_bits(0b00000010100101000001111010011100) == 964176192

def test_reverse_bits_example2():
    assert reverse_bits(0b11111111111111111111111111111101) == 3221225471

def test_reverse_bits_zero():
    assert reverse_bits(0) == 0

def test_reverse_bits_one():
    assert reverse_bits(1) == 2147483648

def test_reverse_bits_all_ones():
    assert reverse_bits(0xFFFFFFFF) == 0xFFFFFFFF
