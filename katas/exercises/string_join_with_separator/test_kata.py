"""Tests for string join with separator kata."""


try:
    from user_kata import join_strings
except ImportError:
    from .reference import join_strings


def test_join_simple():

    assert join_strings(["hello", "world"], " ") == "hello world"
    assert join_strings(["a", "b", "c"], ",") == "a,b,c"


def test_join_empty_list():

    assert join_strings([], " ") == ""


def test_join_single_element():

    assert join_strings(["hello"], " ") == "hello"


def test_join_no_separator():

    assert join_strings(["hello", "world"], "") == "helloworld"


def test_join_multi_char_separator():

    assert join_strings(["hello", "world"], " - ") == "hello - world"
    assert join_strings(["a", "b", "c"], "::") == "a::b::c"


def test_join_empty_strings():

    assert join_strings(["", "", ""], ",") == ",,"
    assert join_strings(["a", "", "c"], ",") == "a,,c"
