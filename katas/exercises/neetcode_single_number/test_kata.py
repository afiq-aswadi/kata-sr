"""Tests for Single Number kata."""

def test_single_number_example1():
    from template import single_number
    assert single_number([2,2,1]) == 1

def test_single_number_example2():
    from template import single_number
    assert single_number([4,1,2,1,2]) == 4

def test_single_number_example3():
    from template import single_number
    assert single_number([1]) == 1

def test_single_number_negative():
    from template import single_number
    assert single_number([-1,-1,2]) == 2

def test_single_number_multiple_pairs():
    from template import single_number
    assert single_number([1,1,2,2,3,3,4]) == 4
