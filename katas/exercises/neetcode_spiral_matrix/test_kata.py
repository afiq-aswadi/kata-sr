"""Tests for Spiral Matrix kata."""

def test_spiral_order_example1():
    from template import spiral_order
    assert spiral_order([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]

def test_spiral_order_example2():
    from template import spiral_order
    assert spiral_order([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) == [1,2,3,4,8,12,11,10,9,5,6,7]

def test_spiral_order_single():
    from template import spiral_order
    assert spiral_order([[1]]) == [1]

def test_spiral_order_single_row():
    from template import spiral_order
    assert spiral_order([[1,2,3,4]]) == [1,2,3,4]

def test_spiral_order_single_col():
    from template import spiral_order
    assert spiral_order([[1],[2],[3]]) == [1,2,3]
