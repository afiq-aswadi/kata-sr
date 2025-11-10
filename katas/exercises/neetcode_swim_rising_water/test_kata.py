"""Tests for Swim in Rising Water kata."""

def test_swim_in_water_example1():
    from template import swim_in_water
    grid = [[0,2],[1,3]]
    assert swim_in_water(grid) == 3

def test_swim_in_water_example2():
    from template import swim_in_water
    grid = [[0,1,2,3,4],[24,23,22,21,5],[12,13,14,15,16],[11,17,18,19,20],[10,9,8,7,6]]
    assert swim_in_water(grid) == 16

def test_swim_in_water_single():
    from template import swim_in_water
    grid = [[0]]
    assert swim_in_water(grid) == 0

def test_swim_in_water_simple():
    from template import swim_in_water
    grid = [[0,1],[2,3]]
    assert swim_in_water(grid) == 3
