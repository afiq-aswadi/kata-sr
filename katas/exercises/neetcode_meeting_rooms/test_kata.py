"""Tests for Meeting Rooms kata."""

def test_can_attend_meetings_example1():
    from template import can_attend_meetings
    assert can_attend_meetings([[0,30],[5,10],[15,20]]) == False

def test_can_attend_meetings_example2():
    from template import can_attend_meetings
    assert can_attend_meetings([[7,10],[2,4]]) == True

def test_can_attend_meetings_empty():
    from template import can_attend_meetings
    assert can_attend_meetings([]) == True

def test_can_attend_meetings_single():
    from template import can_attend_meetings
    assert can_attend_meetings([[1,5]]) == True

def test_can_attend_meetings_adjacent():
    from template import can_attend_meetings
    assert can_attend_meetings([[1,5],[5,10]]) == True
