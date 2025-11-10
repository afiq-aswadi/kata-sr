"""Tests for Plus One kata."""

def test_plus_one_example1():
    from template import plus_one
    assert plus_one([1,2,3]) == [1,2,4]

def test_plus_one_example2():
    from template import plus_one
    assert plus_one([4,3,2,1]) == [4,3,2,2]

def test_plus_one_example3():
    from template import plus_one
    assert plus_one([9]) == [1,0]

def test_plus_one_multiple_nines():
    from template import plus_one
    assert plus_one([9,9,9]) == [1,0,0,0]

def test_plus_one_with_carry():
    from template import plus_one
    assert plus_one([1,9,9]) == [2,0,0]
