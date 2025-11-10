"""Tests for Meeting Rooms II kata."""

def test_min_meeting_rooms_example1():
    from template import min_meeting_rooms
    assert min_meeting_rooms([[0,30],[5,10],[15,20]]) == 2

def test_min_meeting_rooms_example2():
    from template import min_meeting_rooms
    assert min_meeting_rooms([[7,10],[2,4]]) == 1

def test_min_meeting_rooms_single():
    from template import min_meeting_rooms
    assert min_meeting_rooms([[1,5]]) == 1

def test_min_meeting_rooms_all_overlap():
    from template import min_meeting_rooms
    assert min_meeting_rooms([[1,10],[2,10],[3,10]]) == 3

def test_min_meeting_rooms_sequential():
    from template import min_meeting_rooms
    assert min_meeting_rooms([[1,5],[5,10],[10,15]]) == 1
