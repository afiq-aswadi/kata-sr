"""Tests for Missing Number kata."""

def test_missing_number_example1():
    from template import missing_number
    assert missing_number([3,0,1]) == 2

def test_missing_number_example2():
    from template import missing_number
    assert missing_number([0,1]) == 2

def test_missing_number_example3():
    from template import missing_number
    assert missing_number([9,6,4,2,3,5,7,0,1]) == 8

def test_missing_number_zero():
    from template import missing_number
    assert missing_number([1]) == 0

def test_missing_number_last():
    from template import missing_number
    assert missing_number([0,1,2,3]) == 4
