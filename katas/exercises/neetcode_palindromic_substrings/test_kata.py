"""Tests for Palindromic Substrings kata."""

try:
    from user_kata import count_substrings
except ImportError:
    from .reference import count_substrings


def test_count_substrings_example1():
    assert count_substrings("abc") == 3

def test_count_substrings_example2():
    assert count_substrings("aaa") == 6

def test_count_substrings_single():
    assert count_substrings("a") == 1

def test_count_substrings_two():
    assert count_substrings("ab") == 2
    assert count_substrings("aa") == 3
