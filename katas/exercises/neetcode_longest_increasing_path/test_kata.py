"""Tests for Longest Increasing Path in a Matrix kata."""

def test_longest_increasing_path_example1():
    from template import longest_increasing_path
    assert longest_increasing_path([[9,9,4],[6,6,8],[2,1,1]]) == 4

def test_longest_increasing_path_example2():
    from template import longest_increasing_path
    assert longest_increasing_path([[3,4,5],[3,2,6],[2,2,1]]) == 4

def test_longest_increasing_path_example3():
    from template import longest_increasing_path
    assert longest_increasing_path([[1]]) == 1

def test_longest_increasing_path_all_same():
    from template import longest_increasing_path
    assert longest_increasing_path([[1,1],[1,1]]) == 1
