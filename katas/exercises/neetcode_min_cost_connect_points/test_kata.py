"""Tests for Min Cost to Connect All Points kata."""

def test_min_cost_connect_points_example1():
    from template import min_cost_connect_points
    points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
    assert min_cost_connect_points(points) == 20

def test_min_cost_connect_points_example2():
    from template import min_cost_connect_points
    points = [[3,12],[-2,5],[-4,1]]
    assert min_cost_connect_points(points) == 18

def test_min_cost_connect_points_two():
    from template import min_cost_connect_points
    points = [[0,0],[1,1]]
    assert min_cost_connect_points(points) == 2

def test_min_cost_connect_points_line():
    from template import min_cost_connect_points
    points = [[0,0],[1,0],[2,0]]
    assert min_cost_connect_points(points) == 2
