"""Tests for Coin Change kata."""

try:
    from user_kata import coin_change
except ImportError:
    from .reference import coin_change


def test_coin_change_example1():
    assert coin_change([1,2,5], 11) == 3

def test_coin_change_example2():
    assert coin_change([2], 3) == -1

def test_coin_change_example3():
    assert coin_change([1], 0) == 0

def test_coin_change_single_coin():
    assert coin_change([1], 2) == 2

def test_coin_change_impossible():
    assert coin_change([2], 1) == -1
