"""Tests for Min Cost to Connect All Points kata."""

try:
    from user_kata import min_cost_connect_points
except ImportError:
    from .reference import min_cost_connect_points


def test_min_cost_connect_points_example1():
    points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
    assert min_cost_connect_points(points) == 20

def test_min_cost_connect_points_example2():
    points = [[3,12],[-2,5],[-4,1]]
    assert min_cost_connect_points(points) == 18

def test_min_cost_connect_points_two():
    points = [[0,0],[1,1]]
    assert min_cost_connect_points(points) == 2

def test_min_cost_connect_points_line():
    points = [[0,0],[1,0],[2,0]]
    assert min_cost_connect_points(points) == 2
