"""Tests for string palindrome checker kata."""


try:
    from user_kata import is_palindrome
except ImportError:
    from .reference import is_palindrome


def test_palindrome_simple():

    assert is_palindrome("racecar") is True
    assert is_palindrome("hello") is False


def test_palindrome_single_char():

    assert is_palindrome("a") is True


def test_palindrome_empty():

    assert is_palindrome("") is True


def test_palindrome_case_insensitive():

    assert is_palindrome("Racecar") is True
    assert is_palindrome("RaceCar") is True


def test_palindrome_with_spaces():

    assert is_palindrome("race car") is True
    assert is_palindrome("A man a plan a canal Panama") is True


def test_palindrome_non_palindrome():

    assert is_palindrome("python") is False
    assert is_palindrome("hello world") is False
