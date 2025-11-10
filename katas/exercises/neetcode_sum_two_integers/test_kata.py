"""Tests for Sum of Two Integers kata."""

def test_get_sum_example1():
    from template import get_sum
    assert get_sum(1, 2) == 3

def test_get_sum_example2():
    from template import get_sum
    assert get_sum(2, 3) == 5

def test_get_sum_zero():
    from template import get_sum
    assert get_sum(0, 0) == 0

def test_get_sum_negative():
    from template import get_sum
    assert get_sum(-1, 1) == 0

def test_get_sum_both_negative():
    from template import get_sum
    assert get_sum(-2, -3) == -5
