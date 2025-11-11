"""Tests for Burst Balloons kata."""

try:
    from user_kata import max_coins
except ImportError:
    from .reference import max_coins


def test_max_coins_example1():
    assert max_coins([3,1,5,8]) == 167

def test_max_coins_example2():
    assert max_coins([1,5]) == 10

def test_max_coins_single():
    assert max_coins([5]) == 5

def test_max_coins_three():
    assert max_coins([1,2,3]) == 12
