"""Tests for Best Time to Buy and Sell Stock kata."""

def test_max_profit_example1():
    from template import max_profit
    assert max_profit([7,1,5,3,6,4]) == 5

def test_max_profit_example2():
    from template import max_profit
    assert max_profit([7,6,4,3,1]) == 0

def test_max_profit_single():
    from template import max_profit
    assert max_profit([1]) == 0
