"""Tests for string word count kata."""


try:
    from user_kata import count_words
except ImportError:
    from .reference import count_words


def test_count_words_simple():

    assert count_words("hello world") == 2
    assert count_words("one two three") == 3


def test_count_words_single():

    assert count_words("hello") == 1


def test_count_words_empty():

    assert count_words("") == 0


def test_count_words_spaces_only():

    assert count_words("   ") == 0


def test_count_words_multiple_spaces():

    assert count_words("hello  world") == 2
    assert count_words("  hello   world  ") == 2


def test_count_words_leading_trailing_spaces():

    assert count_words("  hello world  ") == 2


def test_count_words_tabs_newlines():

    assert count_words("hello\tworld") == 2
    assert count_words("hello\nworld") == 2
