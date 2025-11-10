"""Tests for Course Schedule II kata."""

def test_find_order_example1():
    from template import find_order
    result = find_order(2, [[1,0]])
    assert result == [0,1]

def test_find_order_example2():
    from template import find_order
    result = find_order(4, [[1,0],[2,0],[3,1],[3,2]])
    # Multiple valid orderings
    assert len(result) == 4
    # Verify ordering is valid
    pos = {course: i for i, course in enumerate(result)}
    assert pos[0] < pos[1] and pos[0] < pos[2]
    assert pos[1] < pos[3] and pos[2] < pos[3]

def test_find_order_example3():
    from template import find_order
    result = find_order(1, [])
    assert result == [0]

def test_find_order_cycle():
    from template import find_order
    result = find_order(2, [[1,0],[0,1]])
    assert result == []
