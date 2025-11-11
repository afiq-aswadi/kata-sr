"""Tests for Walls and Gates kata."""

INF = 2147483647

try:
    from user_kata import walls_and_gates
except ImportError:
    from .reference import walls_and_gates


def test_walls_and_gates_example1():
    rooms = [[INF,-1,0,INF],[INF,INF,INF,-1],[INF,-1,INF,-1],[0,-1,INF,INF]]
    walls_and_gates(rooms)
    expected = [[3,-1,0,1],[2,2,1,-1],[1,-1,2,-1],[0,-1,3,4]]
    assert rooms == expected

def test_walls_and_gates_example2():
    rooms = [[-1]]
    walls_and_gates(rooms)
    expected = [[-1]]
    assert rooms == expected

def test_walls_and_gates_simple():
    rooms = [[0,INF,INF],[INF,-1,INF],[INF,INF,0]]
    walls_and_gates(rooms)
    expected = [[0,1,2],[1,-1,1],[2,1,0]]
    assert rooms == expected

def test_walls_and_gates_single_gate():
    rooms = [[0,INF],[INF,INF]]
    walls_and_gates(rooms)
    expected = [[0,1],[1,2]]
    assert rooms == expected
