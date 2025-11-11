"""Tests for Maximum Product Subarray kata."""

try:
    from user_kata import max_product
except ImportError:
    from .reference import max_product


def test_max_product_example1():
    assert max_product([2,3,-2,4]) == 6

def test_max_product_example2():
    assert max_product([-2,0,-1]) == 0

def test_max_product_all_negative():
    assert max_product([-2,-3,-4]) == 12

def test_max_product_single():
    assert max_product([5]) == 5
    assert max_product([-5]) == -5
