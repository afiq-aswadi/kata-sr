"""Tests for string uppercase conversion kata."""


def test_uppercase_simple():
    from template import to_uppercase

    assert to_uppercase("hello") == "HELLO"
    assert to_uppercase("world") == "WORLD"


def test_uppercase_already_upper():
    from template import to_uppercase

    assert to_uppercase("HELLO") == "HELLO"


def test_uppercase_mixed_case():
    from template import to_uppercase

    assert to_uppercase("HeLLo") == "HELLO"
    assert to_uppercase("PyThOn") == "PYTHON"


def test_uppercase_with_spaces():
    from template import to_uppercase

    assert to_uppercase("hello world") == "HELLO WORLD"


def test_uppercase_with_punctuation():
    from template import to_uppercase

    assert to_uppercase("hello!") == "HELLO!"
    assert to_uppercase("hello, world!") == "HELLO, WORLD!"


def test_uppercase_empty():
    from template import to_uppercase

    assert to_uppercase("") == ""


def test_uppercase_numbers():
    from template import to_uppercase

    assert to_uppercase("abc123") == "ABC123"
