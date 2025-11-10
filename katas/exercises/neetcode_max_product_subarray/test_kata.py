"""Tests for Maximum Product Subarray kata."""

def test_max_product_example1():
    from template import max_product
    assert max_product([2,3,-2,4]) == 6

def test_max_product_example2():
    from template import max_product
    assert max_product([-2,0,-1]) == 0

def test_max_product_all_negative():
    from template import max_product
    assert max_product([-2,-3,-4]) == 12

def test_max_product_single():
    from template import max_product
    assert max_product([5]) == 5
    assert max_product([-5]) == -5
