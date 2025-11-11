"""Tests for Product of Array Except Self kata."""

try:
    from user_kata import product_except_self
except ImportError:
    from .reference import product_except_self


def test_product_except_self_example1():
    assert product_except_self([1,2,3,4]) == [24,12,8,6]

def test_product_except_self_example2():
    assert product_except_self([-1,1,0,-3,3]) == [0,0,9,0,0]

def test_product_except_self_two_elements():
    assert product_except_self([1,2]) == [2,1]
