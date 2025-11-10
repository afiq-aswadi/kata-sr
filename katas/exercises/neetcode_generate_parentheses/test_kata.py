"""Tests for Generate Parentheses kata."""

def test_generate_parenthesis_n3():
    from template import generate_parenthesis
    result = sorted(generate_parenthesis(3))
    expected = sorted(["((()))","(()())","(())()","()(())","()()()"])
    assert result == expected

def test_generate_parenthesis_n1():
    from template import generate_parenthesis
    assert generate_parenthesis(1) == ["()"]
