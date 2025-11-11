"""Tests for string uppercase conversion kata."""


try:
    from user_kata import to_uppercase
except ImportError:
    from .reference import to_uppercase


def test_uppercase_simple():

    assert to_uppercase("hello") == "HELLO"
    assert to_uppercase("world") == "WORLD"


def test_uppercase_already_upper():

    assert to_uppercase("HELLO") == "HELLO"


def test_uppercase_mixed_case():

    assert to_uppercase("HeLLo") == "HELLO"
    assert to_uppercase("PyThOn") == "PYTHON"


def test_uppercase_with_spaces():

    assert to_uppercase("hello world") == "HELLO WORLD"


def test_uppercase_with_punctuation():

    assert to_uppercase("hello!") == "HELLO!"
    assert to_uppercase("hello, world!") == "HELLO, WORLD!"


def test_uppercase_empty():

    assert to_uppercase("") == ""


def test_uppercase_numbers():

    assert to_uppercase("abc123") == "ABC123"
