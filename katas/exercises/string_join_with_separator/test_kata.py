"""Tests for string join with separator kata."""


def test_join_simple():
    from template import join_strings

    assert join_strings(["hello", "world"], " ") == "hello world"
    assert join_strings(["a", "b", "c"], ",") == "a,b,c"


def test_join_empty_list():
    from template import join_strings

    assert join_strings([], " ") == ""


def test_join_single_element():
    from template import join_strings

    assert join_strings(["hello"], " ") == "hello"


def test_join_no_separator():
    from template import join_strings

    assert join_strings(["hello", "world"], "") == "helloworld"


def test_join_multi_char_separator():
    from template import join_strings

    assert join_strings(["hello", "world"], " - ") == "hello - world"
    assert join_strings(["a", "b", "c"], "::") == "a::b::c"


def test_join_empty_strings():
    from template import join_strings

    assert join_strings(["", "", ""], ",") == ",,"
    assert join_strings(["a", "", "c"], ",") == "a,,c"
