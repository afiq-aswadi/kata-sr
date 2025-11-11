"""Tests for Valid Parenthesis String kata."""

try:
    from user_kata import check_valid_string
except ImportError:
    from .reference import check_valid_string


def test_check_valid_string_example1():
    assert check_valid_string("()") == True

def test_check_valid_string_example2():
    assert check_valid_string("(*)") == True

def test_check_valid_string_example3():
    assert check_valid_string("(*))") == True

def test_check_valid_string_invalid():
    assert check_valid_string("(()") == False

def test_check_valid_string_stars():
    assert check_valid_string("(((*") == True
