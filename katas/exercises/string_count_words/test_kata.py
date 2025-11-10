"""Tests for string word count kata."""


def test_count_words_simple():
    from template import count_words

    assert count_words("hello world") == 2
    assert count_words("one two three") == 3


def test_count_words_single():
    from template import count_words

    assert count_words("hello") == 1


def test_count_words_empty():
    from template import count_words

    assert count_words("") == 0


def test_count_words_spaces_only():
    from template import count_words

    assert count_words("   ") == 0


def test_count_words_multiple_spaces():
    from template import count_words

    assert count_words("hello  world") == 2
    assert count_words("  hello   world  ") == 2


def test_count_words_leading_trailing_spaces():
    from template import count_words

    assert count_words("  hello world  ") == 2


def test_count_words_tabs_newlines():
    from template import count_words

    assert count_words("hello\tworld") == 2
    assert count_words("hello\nworld") == 2
