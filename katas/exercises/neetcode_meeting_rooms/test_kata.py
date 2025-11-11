"""Tests for Meeting Rooms kata."""

try:
    from user_kata import can_attend_meetings
except ImportError:
    from .reference import can_attend_meetings


def test_can_attend_meetings_example1():
    assert can_attend_meetings([[0,30],[5,10],[15,20]]) == False

def test_can_attend_meetings_example2():
    assert can_attend_meetings([[7,10],[2,4]]) == True

def test_can_attend_meetings_empty():
    assert can_attend_meetings([]) == True

def test_can_attend_meetings_single():
    assert can_attend_meetings([[1,5]]) == True

def test_can_attend_meetings_adjacent():
    assert can_attend_meetings([[1,5],[5,10]]) == True
