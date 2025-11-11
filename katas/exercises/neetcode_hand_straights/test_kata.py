"""Tests for Hand of Straights kata."""

try:
    from user_kata import is_n_straight_hand
except ImportError:
    from .reference import is_n_straight_hand


def test_is_n_straight_hand_example1():
    assert is_n_straight_hand([1,2,3,6,2,3,4,7,8], 3) == True

def test_is_n_straight_hand_example2():
    assert is_n_straight_hand([1,2,3,4,5], 4) == False

def test_is_n_straight_hand_single_group():
    assert is_n_straight_hand([1,2,3], 3) == True

def test_is_n_straight_hand_impossible():
    assert is_n_straight_hand([1,1,2,2,3,3], 2) == False

def test_is_n_straight_hand_size_one():
    assert is_n_straight_hand([1,2,3,4], 1) == True
