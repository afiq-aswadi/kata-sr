"""Tests for Reverse Integer kata."""

try:
    from user_kata import reverse
except ImportError:
    from .reference import reverse


def test_reverse_example1():
    assert reverse(123) == 321

def test_reverse_example2():
    assert reverse(-123) == -321

def test_reverse_example3():
    assert reverse(120) == 21

def test_reverse_zero():
    assert reverse(0) == 0

def test_reverse_overflow():
    assert reverse(1534236469) == 0
