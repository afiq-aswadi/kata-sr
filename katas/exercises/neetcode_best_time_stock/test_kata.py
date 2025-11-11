"""Tests for Best Time to Buy and Sell Stock kata."""

try:
    from user_kata import max_profit
except ImportError:
    from .reference import max_profit


def test_max_profit_example1():
    assert max_profit([7,1,5,3,6,4]) == 5

def test_max_profit_example2():
    assert max_profit([7,6,4,3,1]) == 0

def test_max_profit_single():
    assert max_profit([1]) == 0
