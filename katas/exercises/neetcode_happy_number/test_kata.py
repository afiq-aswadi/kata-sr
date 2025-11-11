"""Tests for Happy Number kata."""

try:
    from user_kata import is_happy
except ImportError:
    from .reference import is_happy


def test_is_happy_example1():
    assert is_happy(19) == True

def test_is_happy_example2():
    assert is_happy(2) == False

def test_is_happy_one():
    assert is_happy(1) == True

def test_is_happy_seven():
    assert is_happy(7) == True

def test_is_happy_not_happy():
    assert is_happy(4) == False
