"""Tests for Sum of Two Integers kata."""

try:
    from user_kata import get_sum
except ImportError:
    from .reference import get_sum


def test_get_sum_example1():
    assert get_sum(1, 2) == 3

def test_get_sum_example2():
    assert get_sum(2, 3) == 5

def test_get_sum_zero():
    assert get_sum(0, 0) == 0

def test_get_sum_negative():
    assert get_sum(-1, 1) == 0

def test_get_sum_both_negative():
    assert get_sum(-2, -3) == -5
