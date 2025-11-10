"""Tests for Evaluate Reverse Polish Notation kata."""

def test_eval_rpn_example1():
    from template import eval_rpn
    assert eval_rpn(["2","1","+","3","*"]) == 9

def test_eval_rpn_example2():
    from template import eval_rpn
    assert eval_rpn(["4","13","5","/","+"]) == 6

def test_eval_rpn_single():
    from template import eval_rpn
    assert eval_rpn(["42"]) == 42
