"""Tests for Product of Array Except Self kata."""

def test_product_except_self_example1():
    from template import product_except_self
    assert product_except_self([1,2,3,4]) == [24,12,8,6]

def test_product_except_self_example2():
    from template import product_except_self
    assert product_except_self([-1,1,0,-3,3]) == [0,0,9,0,0]

def test_product_except_self_two_elements():
    from template import product_except_self
    assert product_except_self([1,2]) == [2,1]
