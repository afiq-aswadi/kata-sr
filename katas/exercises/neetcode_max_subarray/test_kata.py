"""Tests for Maximum Subarray kata."""

def test_max_subarray_example1():
    from template import max_subarray
    assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6

def test_max_subarray_example2():
    from template import max_subarray
    assert max_subarray([1]) == 1

def test_max_subarray_example3():
    from template import max_subarray
    assert max_subarray([5,4,-1,7,8]) == 23

def test_max_subarray_all_negative():
    from template import max_subarray
    assert max_subarray([-2,-1]) == -1

def test_max_subarray_mixed():
    from template import max_subarray
    assert max_subarray([-2,1]) == 1
