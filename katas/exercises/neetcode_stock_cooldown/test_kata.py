"""Tests for Best Time to Buy and Sell Stock with Cooldown kata."""

def test_max_profit_example1():
    from template import max_profit
    assert max_profit([1,2,3,0,2]) == 3

def test_max_profit_example2():
    from template import max_profit
    assert max_profit([1]) == 0

def test_max_profit_ascending():
    from template import max_profit
    assert max_profit([1,2,3,4,5]) == 4

def test_max_profit_descending():
    from template import max_profit
    assert max_profit([5,4,3,2,1]) == 0
