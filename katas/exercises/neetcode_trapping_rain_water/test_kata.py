"""Tests for Trapping Rain Water kata."""

def test_trap_example1():
    from template import trap
    assert trap([0,1,0,2,1,0,1,3,2,1,2,1]) == 6

def test_trap_example2():
    from template import trap
    assert trap([4,2,0,3,2,5]) == 9

def test_trap_no_water():
    from template import trap
    assert trap([1,2,3,4,5]) == 0
