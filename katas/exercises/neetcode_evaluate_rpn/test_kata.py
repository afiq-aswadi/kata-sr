"""Tests for Evaluate Reverse Polish Notation kata."""

try:
    from user_kata import eval_rpn
except ImportError:
    from .reference import eval_rpn


def test_eval_rpn_example1():
    assert eval_rpn(["2","1","+","3","*"]) == 9

def test_eval_rpn_example2():
    assert eval_rpn(["4","13","5","/","+"]) == 6

def test_eval_rpn_single():
    assert eval_rpn(["42"]) == 42
