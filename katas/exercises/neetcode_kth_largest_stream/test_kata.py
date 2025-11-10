"""Tests for Kth Largest Element in a Stream kata."""

def test_kth_largest_stream():
    from template import KthLargest

    kth = KthLargest(3, [4, 5, 8, 2])
    assert kth.add(3) == 4
    assert kth.add(5) == 5
    assert kth.add(10) == 5
    assert kth.add(9) == 8
    assert kth.add(4) == 8

def test_kth_largest_stream_single():
    from template import KthLargest

    kth = KthLargest(1, [])
    assert kth.add(-3) == -3
    assert kth.add(-2) == -2
    assert kth.add(-4) == -2
    assert kth.add(0) == 0
    assert kth.add(4) == 4

def test_kth_largest_stream_duplicates():
    from template import KthLargest

    kth = KthLargest(2, [0])
    assert kth.add(-1) == -1
    assert kth.add(1) == 0
    assert kth.add(-2) == 0
    assert kth.add(-4) == 0
    assert kth.add(3) == 1
