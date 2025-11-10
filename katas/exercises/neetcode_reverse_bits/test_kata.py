"""Tests for Reverse Bits kata."""

def test_reverse_bits_example1():
    from template import reverse_bits
    assert reverse_bits(0b00000010100101000001111010011100) == 964176192

def test_reverse_bits_example2():
    from template import reverse_bits
    assert reverse_bits(0b11111111111111111111111111111101) == 3221225471

def test_reverse_bits_zero():
    from template import reverse_bits
    assert reverse_bits(0) == 0

def test_reverse_bits_one():
    from template import reverse_bits
    assert reverse_bits(1) == 2147483648

def test_reverse_bits_all_ones():
    from template import reverse_bits
    assert reverse_bits(0xFFFFFFFF) == 0xFFFFFFFF
