"""Tests for Generate Parentheses kata."""

try:
    from user_kata import generate_parenthesis
except ImportError:
    from .reference import generate_parenthesis


def test_generate_parenthesis_n3():
    result = sorted(generate_parenthesis(3))
    expected = sorted(["((()))","(()())","(())()","()(())","()()()"])
    assert result == expected

def test_generate_parenthesis_n1():
    assert generate_parenthesis(1) == ["()"]
