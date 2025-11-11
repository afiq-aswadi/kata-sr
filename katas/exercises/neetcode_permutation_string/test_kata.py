"""Tests for Permutation in String kata."""

try:
    from user_kata import check_inclusion
except ImportError:
    from .reference import check_inclusion


def test_check_inclusion_example1():
    assert check_inclusion("ab", "eidbaooo") == True

def test_check_inclusion_example2():
    assert check_inclusion("ab", "eidboaoo") == False

def test_check_inclusion_exact_match():
    assert check_inclusion("abc", "bca") == True
