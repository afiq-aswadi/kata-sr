"""Tests for Hand of Straights kata."""

def test_is_n_straight_hand_example1():
    from template import is_n_straight_hand
    assert is_n_straight_hand([1,2,3,6,2,3,4,7,8], 3) == True

def test_is_n_straight_hand_example2():
    from template import is_n_straight_hand
    assert is_n_straight_hand([1,2,3,4,5], 4) == False

def test_is_n_straight_hand_single_group():
    from template import is_n_straight_hand
    assert is_n_straight_hand([1,2,3], 3) == True

def test_is_n_straight_hand_impossible():
    from template import is_n_straight_hand
    assert is_n_straight_hand([1,1,2,2,3,3], 2) == False

def test_is_n_straight_hand_size_one():
    from template import is_n_straight_hand
    assert is_n_straight_hand([1,2,3,4], 1) == True
