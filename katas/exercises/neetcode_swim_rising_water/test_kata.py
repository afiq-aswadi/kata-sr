"""Tests for Swim in Rising Water kata."""

try:
    from user_kata import swim_in_water
except ImportError:
    from .reference import swim_in_water


def test_swim_in_water_example1():
    grid = [[0,2],[1,3]]
    assert swim_in_water(grid) == 3

def test_swim_in_water_example2():
    grid = [[0,1,2,3,4],[24,23,22,21,5],[12,13,14,15,16],[11,17,18,19,20],[10,9,8,7,6]]
    assert swim_in_water(grid) == 16

def test_swim_in_water_single():
    grid = [[0]]
    assert swim_in_water(grid) == 0

def test_swim_in_water_simple():
    grid = [[0,1],[2,3]]
    assert swim_in_water(grid) == 3
