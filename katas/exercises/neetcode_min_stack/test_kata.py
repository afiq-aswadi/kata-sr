"""Tests for Min Stack kata."""

try:
    from user_kata import MinStack
except ImportError:
    from .reference import MinStack


def test_min_stack():

    stack = MinStack()
    stack.push(-2)
    stack.push(0)
    stack.push(-3)
    assert stack.get_min() == -3
    stack.pop()
    assert stack.top() == 0
    assert stack.get_min() == -2
