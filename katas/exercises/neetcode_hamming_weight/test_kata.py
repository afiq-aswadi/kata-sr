"""Tests for Number of 1 Bits kata."""

try:
    from user_kata import hamming_weight
except ImportError:
    from .reference import hamming_weight


def test_hamming_weight_example1():
    assert hamming_weight(11) == 3

def test_hamming_weight_example2():
    assert hamming_weight(128) == 1

def test_hamming_weight_example3():
    assert hamming_weight(2147483645) == 30

def test_hamming_weight_one():
    assert hamming_weight(1) == 1

def test_hamming_weight_power_of_two():
    assert hamming_weight(16) == 1
