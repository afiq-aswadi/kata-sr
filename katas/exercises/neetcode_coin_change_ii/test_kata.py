"""Tests for Coin Change II kata."""

def test_change_example1():
    from template import change
    assert change(5, [1,2,5]) == 4

def test_change_example2():
    from template import change
    assert change(3, [2]) == 0

def test_change_example3():
    from template import change
    assert change(10, [10]) == 1

def test_change_zero():
    from template import change
    assert change(0, [1,2,5]) == 1

def test_change_small():
    from template import change
    assert change(2, [1,2]) == 2
