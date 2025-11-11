"""Tests for Koko Eating Bananas kata."""

try:
    from user_kata import min_eating_speed
except ImportError:
    from .reference import min_eating_speed


def test_koko_example1():
    assert min_eating_speed([3,6,7,11], 8) == 4

def test_koko_example2():
    assert min_eating_speed([30,11,23,4,20], 5) == 30

def test_koko_example3():
    assert min_eating_speed([30,11,23,4,20], 6) == 23

def test_koko_single_pile():
    assert min_eating_speed([1000000000], 2) == 500000000

def test_koko_exact_hours():
    assert min_eating_speed([1,1,1,1], 4) == 1
