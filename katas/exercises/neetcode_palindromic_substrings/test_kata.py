"""Tests for Palindromic Substrings kata."""

def test_count_substrings_example1():
    from template import count_substrings
    assert count_substrings("abc") == 3

def test_count_substrings_example2():
    from template import count_substrings
    assert count_substrings("aaa") == 6

def test_count_substrings_single():
    from template import count_substrings
    assert count_substrings("a") == 1

def test_count_substrings_two():
    from template import count_substrings
    assert count_substrings("ab") == 2
    assert count_substrings("aa") == 3
