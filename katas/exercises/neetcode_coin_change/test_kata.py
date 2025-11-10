"""Tests for Coin Change kata."""

def test_coin_change_example1():
    from template import coin_change
    assert coin_change([1,2,5], 11) == 3

def test_coin_change_example2():
    from template import coin_change
    assert coin_change([2], 3) == -1

def test_coin_change_example3():
    from template import coin_change
    assert coin_change([1], 0) == 0

def test_coin_change_single_coin():
    from template import coin_change
    assert coin_change([1], 2) == 2

def test_coin_change_impossible():
    from template import coin_change
    assert coin_change([2], 1) == -1
