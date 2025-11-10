"""Tests for string palindrome checker kata."""


def test_palindrome_simple():
    from template import is_palindrome

    assert is_palindrome("racecar") is True
    assert is_palindrome("hello") is False


def test_palindrome_single_char():
    from template import is_palindrome

    assert is_palindrome("a") is True


def test_palindrome_empty():
    from template import is_palindrome

    assert is_palindrome("") is True


def test_palindrome_case_insensitive():
    from template import is_palindrome

    assert is_palindrome("Racecar") is True
    assert is_palindrome("RaceCar") is True


def test_palindrome_with_spaces():
    from template import is_palindrome

    assert is_palindrome("race car") is True
    assert is_palindrome("A man a plan a canal Panama") is True


def test_palindrome_non_palindrome():
    from template import is_palindrome

    assert is_palindrome("python") is False
    assert is_palindrome("hello world") is False
