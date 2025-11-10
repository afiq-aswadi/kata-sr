"""Tests for Number of 1 Bits kata."""

def test_hamming_weight_example1():
    from template import hamming_weight
    assert hamming_weight(11) == 3

def test_hamming_weight_example2():
    from template import hamming_weight
    assert hamming_weight(128) == 1

def test_hamming_weight_example3():
    from template import hamming_weight
    assert hamming_weight(2147483645) == 30

def test_hamming_weight_one():
    from template import hamming_weight
    assert hamming_weight(1) == 1

def test_hamming_weight_power_of_two():
    from template import hamming_weight
    assert hamming_weight(16) == 1
