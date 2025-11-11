"""Tests for string reversal kata."""

try:
    from user_kata import reverse_string
except ImportError:
    from .reference import reverse_string


def test_reverse_simple():
    assert reverse_string("hello") == "olleh"
    assert reverse_string("world") == "dlrow"


def test_reverse_empty():
    assert reverse_string("") == ""


def test_reverse_single_char():
    assert reverse_string("a") == "a"


def test_reverse_palindrome():
    assert reverse_string("racecar") == "racecar"


def test_reverse_with_spaces():
    assert reverse_string("hello world") == "dlrow olleh"


def test_reverse_with_special_chars():
    assert reverse_string("hello!") == "!olleh"
    assert reverse_string("a@b#c") == "c#b@a"
