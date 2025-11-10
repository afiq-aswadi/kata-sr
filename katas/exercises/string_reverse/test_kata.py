"""Tests for string reversal kata."""


def test_reverse_simple():
    from template import reverse_string

    assert reverse_string("hello") == "olleh"
    assert reverse_string("world") == "dlrow"


def test_reverse_empty():
    from template import reverse_string

    assert reverse_string("") == ""


def test_reverse_single_char():
    from template import reverse_string

    assert reverse_string("a") == "a"


def test_reverse_palindrome():
    from template import reverse_string

    assert reverse_string("racecar") == "racecar"


def test_reverse_with_spaces():
    from template import reverse_string

    assert reverse_string("hello world") == "dlrow olleh"


def test_reverse_with_special_chars():
    from template import reverse_string

    assert reverse_string("hello!") == "!olleh"
    assert reverse_string("a@b#c") == "c#b@a"
