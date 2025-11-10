"""Tests for Pacific Atlantic Water Flow kata."""

def test_pacific_atlantic_example1():
    from template import pacific_atlantic
    heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
    result = pacific_atlantic(heights)
    expected = [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
    assert sorted(result) == sorted(expected)

def test_pacific_atlantic_example2():
    from template import pacific_atlantic
    heights = [[1]]
    result = pacific_atlantic(heights)
    assert result == [[0,0]]

def test_pacific_atlantic_simple():
    from template import pacific_atlantic
    heights = [[1,2],[2,1]]
    result = pacific_atlantic(heights)
    expected = [[0,0],[0,1],[1,0],[1,1]]
    assert sorted(result) == sorted(expected)

def test_pacific_atlantic_diagonal():
    from template import pacific_atlantic
    heights = [[1,2,3],[8,9,4],[7,6,5]]
    result = pacific_atlantic(heights)
    expected = [[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
    assert sorted(result) == sorted(expected)
