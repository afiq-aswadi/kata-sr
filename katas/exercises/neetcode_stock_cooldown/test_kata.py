"""Tests for Best Time to Buy and Sell Stock with Cooldown kata."""

try:
    from user_kata import max_profit
except ImportError:
    from .reference import max_profit


def test_max_profit_example1():
    assert max_profit([1,2,3,0,2]) == 3

def test_max_profit_example2():
    assert max_profit([1]) == 0

def test_max_profit_ascending():
    assert max_profit([1,2,3,4,5]) == 4

def test_max_profit_descending():
    assert max_profit([5,4,3,2,1]) == 0
