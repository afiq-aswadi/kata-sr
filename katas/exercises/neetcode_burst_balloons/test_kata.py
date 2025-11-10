"""Tests for Burst Balloons kata."""

def test_max_coins_example1():
    from template import max_coins
    assert max_coins([3,1,5,8]) == 167

def test_max_coins_example2():
    from template import max_coins
    assert max_coins([1,5]) == 10

def test_max_coins_single():
    from template import max_coins
    assert max_coins([5]) == 5

def test_max_coins_three():
    from template import max_coins
    assert max_coins([1,2,3]) == 12
